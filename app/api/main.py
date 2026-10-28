import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ApiException

from app.api.dependencies import get_agent, get_ingestion, get_pipeline, get_retriever, get_store
from app.api.schemas import AskRequest, AskResponse, DeleteResponse, DocumentInfo, HealthResponse
from app.core.config import get_settings
from app.core.log import setup_logging
from app.ingestion.service import PROTECTED_SOURCES, IngestionError, IngestionService, IngestResult
from app.llm.client import LLMError
from app.rag.agent import InvestigationAgent
from app.rag.pipeline import RAGPipeline
from app.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # Loads the embedding model at startup instead of during the first question
    get_retriever()
    yield


app = FastAPI(title="AI Engine", lifespan=lifespan)


@app.exception_handler(LLMError)
async def llm_error(request: Request, error: LLMError) -> JSONResponse:
    # LLMError messages are written to be shown to the user
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(ApiException)
async def vector_db_error(request: Request, error: ApiException) -> JSONResponse:
    logger.error("Qdrant request failed: %s", error)
    return JSONResponse(status_code=503, content={"detail": "The vector database is unavailable"})


@app.exception_handler(IngestionError)
async def ingestion_error(request: Request, error: IngestionError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


def _file_name(filename: str | None) -> str:
    # Only the last part of the name is kept: a client can send "../../app/main.py".
    # Backslashes count as separators too, also on Linux where Path would keep them
    name = PurePosixPath((filename or "").replace("\\", "/")).name
    if name in ("", ".", ".."):
        raise HTTPException(status_code=400, detail="The file needs a name")
    return name


@app.get("/health")
def health(store: Annotated[QdrantStore, Depends(get_store)]) -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        collection=store.collection,
        chunks=store.count(),
        llm_model=settings.groq_model,
        embedding_model=settings.embedding_model,
    )


# Plain `def` routes: FastAPI runs them in a thread pool, so a slow LLM call does not block other requests
@app.post("/ask")
def ask(
    body: AskRequest,
    agent: Annotated[InvestigationAgent, Depends(get_agent)],
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
) -> AskResponse:
    started = time.perf_counter()
    if body.method == "agent":
        answer = agent.answer(body.question, mode=body.search_mode)
    else:
        answer = pipeline.answer(body.question, mode=body.search_mode)
    duration_ms = round((time.perf_counter() - started) * 1000)
    logger.info("Answered with %s/%s in %d ms", body.method, body.search_mode, duration_ms)
    return AskResponse.from_answer(answer, duration_ms)


@app.get("/documents")
def list_documents(store: Annotated[QdrantStore, Depends(get_store)]) -> list[DocumentInfo]:
    return [
        DocumentInfo(source=source, chunks=chunks, protected=source in PROTECTED_SOURCES)
        for source, chunks in sorted(store.list_sources().items())
    ]


# Uploading a file with the name of an indexed one replaces it (see IngestionService)
@app.post("/documents", status_code=201)
def upload_document(
    file: UploadFile, service: Annotated[IngestionService, Depends(get_ingestion)]
) -> IngestResult:
    name = _file_name(file.filename)
    # Reading one byte past the limit tells "exactly at the limit" from "too large"
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Files over {MAX_UPLOAD_BYTES // 2**20} MB are not accepted")

    # The parsers read from disk, so the upload is saved under its own name in a temporary folder
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / name
        path.write_bytes(content)
        return service.ingest_file(path)


@app.delete("/documents/{source}")
def delete_document(
    source: str, service: Annotated[IngestionService, Depends(get_ingestion)]
) -> DeleteResponse:
    deleted = service.delete(source)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"{source} is not in the collection")
    return DeleteResponse(source=source, deleted_chunks=deleted)
