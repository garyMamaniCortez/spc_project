# Imagen base oficial de Python 3.12.10
# Compatible con TensorFlow, XGBoost, DVC y MLflow
FROM python:3.12.10-slim

# Evitar prompts interactivos y optimizar Python y DVC en Docker
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DVC_NO_ANALYTICS=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema y configurar Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/* \
    && git config --global --add safe.directory /app

# Copiar e instalar dependencias del proyecto
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    grep -v '^-e' /tmp/requirements.txt > /tmp/reqs.txt && \
    pip install -r /tmp/reqs.txt && \
    rm -rf /tmp/requirements.txt /tmp/reqs.txt

# Copiar el codigo completo del proyecto
COPY . /app

# Asegurar permisos de ejecucion y finales de linea UNIX para el script de inicio
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# Instalar el modulo del proyecto en modo editable
RUN pip install -e .

# Exponer el puerto 8080 para MLflow UI
EXPOSE 8080

# Comando por defecto: ejecuta el entrypoint con el flujo de verificación y pipeline
CMD ["/app/docker-entrypoint.sh"]