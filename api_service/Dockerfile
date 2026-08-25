FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copiar el código del servicio FastAPI
COPY . .

# Exponer el puerto 8000
EXPOSE 8000

# Ejecución automática de FastAPI con Uvicorn al arrancar el contenedor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
