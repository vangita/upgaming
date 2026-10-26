import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# BGE models were trained with this instruction in front of short search queries.
# Passages are embedded without it.
QUERY_INSTRUCTIONS = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Embedder:
    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model %s", model_name)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        self.query_instruction = QUERY_INSTRUCTIONS.get(model_name, "")

    def embed_documents(
        self, texts: list[str], batch_size: int = 16, show_progress: bool = False
    ) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            query,
            prompt=self.query_instruction,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(get_settings().embedding_model)
