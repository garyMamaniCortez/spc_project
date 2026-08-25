FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias de sistema requeridas para XGBoost / OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copiar el código del servicio de modelos
COPY . .

# Exponer el puerto interno 5000
EXPOSE 5000

# Ejecutar el servidor de modelos automáticamente
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
