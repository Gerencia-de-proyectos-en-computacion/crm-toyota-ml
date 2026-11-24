FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY data ./data

# Genera dataset calibrado, entrena el RandomForestRegressor (.joblib + metrics.json) y verifica Pytest al construir imagen
RUN python -m scripts.generate_dataset && \
    python -m scripts.train_model && \
    python -m pytest -q

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
