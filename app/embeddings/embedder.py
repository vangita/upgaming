import logging
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# BGE models were trained with this instruction in front of short search queries.
# Passages are embedded without it.
QUERY_INSTRUCTIONS = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
# A fixed revision of the model repository, so a later update of the model cannot change the vectors
REVISIONS = {
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
}
# The model's max_seq_length (sentence_bert_config.json); longer texts are cut
MAX_TOKENS = 512


def download_model(model_name: str) -> tuple[str, str]:
    """Paths of the model's ONNX export and its tokenizer, downloaded once into the Hugging Face cache."""
    revision = REVISIONS.get(model_name)
    model_path = hf_hub_download(model_name, "onnx/model.onnx", revision=revision)
    tokenizer_path = hf_hub_download(model_name, "tokenizer.json", revision=revision)
    return model_path, tokenizer_path


class Embedder:
    """Runs the model's ONNX export with onnxruntime: the vectors of sentence-transformers without torch,
    which alone takes more RAM than a 512 MB free server has."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model %s", model_name)
        model_path, tokenizer_path = download_model(model_name)
        self.model_name = model_name
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_names = {model_input.name for model_input in self.session.get_inputs()}
        self.dimension = self.session.get_outputs()[0].shape[-1]
        self.query_instruction = QUERY_INSTRUCTIONS.get(model_name, "")

    def _embed(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        inputs = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        # Only the inputs this export takes: not every ONNX export has token_type_ids
        hidden = self.session.run(None, {name: value for name, value in inputs.items() if name in self.input_names})[0]
        # The model's pooling (1_Pooling/config.json) is the [CLS] token, then the vector gets unit length
        cls = hidden[:, 0]
        return cls / np.linalg.norm(cls, axis=1, keepdims=True)

    def embed_documents(
        self, texts: list[str], batch_size: int = 16, show_progress: bool = False
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        # A batch is padded to its longest text: texts of similar length go together, longest first
        # (as sentence-transformers does), and the vectors are put back in the original order
        order = np.argsort([-len(text) for text in texts], kind="stable")
        batches = []
        for start in range(0, len(texts), batch_size):
            batches.append(self._embed([texts[i] for i in order[start:start + batch_size]]))
            if show_progress and len(batches) % 200 == 0:
                logger.info("Embedded %d of %d texts", start + batch_size, len(texts))
        vectors = np.empty((len(texts), self.dimension), dtype=np.float32)
        vectors[order] = np.concatenate(batches)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([self.query_instruction + query])[0]


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(get_settings().embedding_model)
