FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_CACHE=/app/models HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY embeddings.py ./embeddings.py
RUN python embeddings.py
COPY app.py smart_search.py ./
RUN useradd --uid 10001 --create-home findit && chown -R findit:findit /app
USER findit
ENV HF_HUB_OFFLINE=1
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
