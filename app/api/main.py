import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ApiException

from app.api.dependencies import get_agent, get_pipeline, get_retriever, get_store
from app.api.schemas import AskRequest, AskResponse, HealthResponse
from app.core.config import get_settings
from app.core.log import setup_logging
from app.llm.client import LLMError
from app.rag.agent import InvestigationAgent
from app.rag.pipeline import RAGPipeline
from app.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # Loads the embedding model at startup instead of during the first question
    get_retriever()
    yield


app = FastAPI(title="AI Investigation Engine", lifespan=lifespan)


@app.exception_handler(LLMError)
async def llm_error(request: Request, error: LLMError) -> JSONResponse:
    # LLMError messages are written to be shown to the user
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(ApiException)
async def vector_db_error(request: Request, error: ApiException) -> JSONResponse:
    logger.error("Qdrant request failed: %s", error)
    return JSONResponse(status_code=503, content={"detail": "The vector database is unavailable"})


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
