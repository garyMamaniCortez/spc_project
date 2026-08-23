# 🐳 Guía de Uso de Docker — spc_project

Este contenedor de Docker ejecuta de manera automatizada todo el pipeline de Machine Learning (**limpieza de datos**, **generación de tabla minable**, **entrenamiento de modelos XGBoost y Red Neuronal MLP**) y levanta el servidor interactivo de **MLflow UI** en el puerto **8080**.

---

## 📋 Requisito Previo: Dataset (`salary.csv`)

Por motivos de seguridad y compatibilidad, la autenticación interactiva de Google Drive se realiza en tu **máquina anfitriona (host)** mediante DVC antes de iniciar el contenedor.

El contenedor verificará la presencia de `data/raw/salary.csv`. Si no lo encuentra, **se detendrá de inmediato** indicándote los pasos a seguir.

### 📥 ¿Cómo obtener el dataset en tu máquina host?

Elige una de las siguientes opciones según tu caso:

#### Opción 1: Si ya tienes DVC instalado
Ejecuta en tu terminal local (PowerShell, CMD o Bash):
```bash
dvc pull
```
*(Si estás en Windows usando el entorno virtual del proyecto: `.\.venv-dvc\Scripts\dvc.exe pull`)*

> 💡 **Nota:** La primera vez, DVC abrirá automáticamente una pestaña en tu navegador para que inicies sesión en Google con la cuenta autorizada.

#### Opción 2: Si NO tienes DVC instalado
Instala DVC con el plugin de Google Drive y descarga el dataset:
```bash
pip install dvc dvc-gdrive
dvc pull
```

#### Opción 3: Colocación manual
Si ya posees el archivo `salary.csv`, simplemente colócalo en la carpeta:
```text
spc_project/data/raw/salary.csv
```

---

## 🚀 Iniciar el Contenedor

Una vez que el archivo `data/raw/salary.csv` se encuentre en tu máquina host, levanta el proyecto con:

```bash
docker compose up
```

*(O para reconstruir la imagen tras cambios en el código: `docker compose up --build`)*

### ⚙️ ¿Qué ejecuta automáticamente el contenedor?

1. 🔍 **Verificación de datos:** Confirma la existencia de `data/raw/salary.csv`.
2. 📊 **Limpieza de datos:** Ejecuta `spc_module.dataset` (`salary.csv` ➔ `data/interim/salary_clean.csv`).
3. ⚙️ **Tabla Minable:** Ejecuta `spc_module.features` (One-hot encoding, escalado numérico y split Train/Test en `data/processed/`).
4. 🤖 **Entrenamiento:** Ejecuta `spc_module.modeling.train` (Entrena **XGBoost** y **Red Neuronal MLP**, guardando métricas, parámetros y artefactos).
5. 🌐 **Servidor MLflow:** Levanta MLflow UI en el puerto **8080**.

---

## 📊 Visualizar Experimentos y Métricas en MLflow

Abre tu navegador en:
👉 **[http://localhost:8080](http://localhost:8080)**

Aquí podrás comparar los modelos, revisar matrices de confusión, curvas ROC y ver los parámetros registrados.

---

## 🛑 Detener el Contenedor

Presiona `Ctrl + C` en tu terminal o ejecuta en otra pestaña:
```bash
docker compose down
```

---

## 🛠️ Comandos de Desarrollo Avanzado

```bash
# Reconstruir la imagen de Docker
docker compose build

# Abrir una terminal interactiva dentro del contenedor
docker compose run --rm app bash

# Ejecutar únicamente la inferencia / predicción
docker compose run --rm app python -m spc_module.modeling.predict

# Ejecutar pruebas unitarias
docker compose run --rm app pytest
```
