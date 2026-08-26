FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENVIRONMENT=docker

WORKDIR /app

# Dependencias de sistema requeridas para XGBoost / OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
# (contexto de build = raíz del proyecto, ver docker-compose.yml)
COPY model_service/requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copiar el código del servicio de modelos
COPY model_service/ .

# Copiar spc_module: necesario porque models/model.pkl serializa clases
# personalizadas (XGBoostModel, StandardNumericalScaler) definidas ahí;
# pickle.load() necesita poder importar ese paquete para reconstruirlas.
COPY spc_module/ ./spc_module/

# Exponer el puerto interno 5000
EXPOSE 5000

# Ejecutar el servidor de modelos automáticamente
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
