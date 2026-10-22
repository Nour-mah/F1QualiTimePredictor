FROM python:3.12-slim

# libgomp1 : bibliothèque de calcul parallèle dont XGBoost a besoin, absente de l'image slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances avant le code : tant que requirements.txt ne change pas, Docker réutilise cette couche en cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Les modules s'importent entre eux depuis src/ (data.*, models.*, api.*)
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

# Par défaut : l'API. Les scripts du pipeline se lancent avec "docker compose run pipeline ..."
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
