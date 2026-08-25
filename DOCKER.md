# 🐳 Guía de Uso de Docker — Arquitectura de Microservicios (spc_project)

Este proyecto implementa una arquitectura modular de producción compuesta por **dos contenedores independientes** comunicados a través de una red privada virtual de Docker (`spc-network`).

```text
       ┌───────────────────────────────┐
       │     Cliente / Navegador       │
       │  http://localhost:8000/docs   │
       └──────────────┬────────────────┘
                      │ Puerto 8000
                      ▼
       ┌───────────────────────────────┐
       │      spc_fastapi (API)        │
       │    (Gateway FastAPI + Pydantic)│
       └──────────────┬────────────────┘
                      │ Red Docker (spc-network)
                      │ HTTP POST http://modelos:5000/predict
                      ▼
       ┌───────────────────────────────┐
       │      spc_modelos (Modelos)    │
       │  (Servicio ML XGBoost / MLP)  │
       └───────────────────────────────┘
```

---

## 🏗️ Estructura de los Servicios

| Servicio | Contenedor | Carpeta | Puerto Interno | Puerto Host | Descripción |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`fastapi`** | `spc_fastapi` | `./api_service/` | `8000` | `8000` | Gateway FastAPI para validación de esquemas (Pydantic V2), Swagger UI y enrutamiento hacia el servicio de modelos. |
| **`modelos`** | `spc_modelos` | `./model_service/` | `5000` | `5000` | Microservicio interno de inferencia con modelos entrenados (XGBoost / Scaler / Features). |

---

## 🚀 Inicio Rápido con Docker Compose

Para construir las imágenes y levantar ambos contenedores de forma automática:

```bash
docker compose up --build
```

Al ejecutar este comando:
1. Docker construye la imagen del contenedor **FastAPI** (`./api_service/Dockerfile`).
2. Docker construye la imagen del contenedor de **modelos** (`./model_service/Dockerfile`).
3. Se crea automáticamente la red compartida `spc_project_spc-network`.
4. Ambos servicios inician automáticamente con Uvicorn (`--host 0.0.0.0`).
5. **FastAPI** queda accesible en tu navegador en:
   👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🌐 Red Docker y Comunicación Interna

* El contenedor `fastapi` se comunica con el contenedor `modelos` utilizando la variable de entorno `MODEL_SERVICE_URL=http://modelos:5000`.
* Docker resuelve el nombre de host `modelos` automáticamente mediante el DNS interno de la red compartida, **sin utilizar `localhost`**.

---

## 🧪 Verificación y Pruebas

### 1. Documentación Interactiva (Swagger UI)
Abre en tu navegador:
* **[http://localhost:8000/docs](http://localhost:8000/docs)**

### 2. Estado y Health Check
* Gateway FastAPI:
  ```bash
  curl http://localhost:8000/
  ```
* Health Check conjunto:
  ```bash
  curl http://localhost:8000/health
  ```

### 3. Inferencia de Prueba (POST `/predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [
         {
           "age": 39,
           "workclass": "State-gov",
           "education-num": 13,
           "marital-status": "Never-married",
           "occupation": "Adm-clerical",
           "relationship": "Not-in-family",
           "race": "White",
           "sex": "Male",
           "capital-gain": 2174,
           "capital-loss": 0,
           "hours-per-week": 40,
           "native-country": "United-States"
         }
       ]
     }'
```

---

## 🛑 Detener los Contenedores

Presiona `Ctrl + C` en tu terminal o ejecuta:
```bash
docker compose down
```
