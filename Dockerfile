# The backend (FastAPI) as deployed on Render. The Streamlit UI is deployed on its own
# (ui/requirements.txt) and reaches this API over HTTP.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home app
WORKDIR /home/app/backend

COPY requirements.txt .
RUN pip install -r requirements.txt

# The app runs without root rights
USER app
COPY --chown=app app/ app/

# The embedding model goes into the image: a cold start does not download it, and at run time
# the app never contacts the Hugging Face Hub
RUN python -c "from app.core.config import Settings; from app.embeddings.embedder import download_model; \
download_model(Settings.model_fields['embedding_model'].default)"
ENV HF_HUB_OFFLINE=1

# Render sets PORT; 8000 elsewhere. exec: uvicorn gets the stop signal itself
CMD ["sh", "-c", "exec uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
