#!/usr/bin/env bash
set -e

# Configurar git safe.directory para evitar advertencias de permisos en volúmenes
git config --global --add safe.directory /app 2>/dev/null || true

# 1. Comprobar si existe el dataset salary.csv
if [ ! -f "data/raw/salary.csv" ]; then
    echo ""
    echo "================================================================================"
    echo "❌ ERROR: DATASET NO ENCONTRADO (data/raw/salary.csv)"
    echo "================================================================================"
    echo ""
    echo "El contenedor requiere el archivo 'data/raw/salary.csv' para ejecutar el pipeline."
    echo "Por razones de seguridad y compatibilidad, la autenticación de Google Drive debe"
    echo "realizarse directamente en tu máquina anfitriona (host)."
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "📖 MANUAL DE INSTRUCCIONES: ¿CÓMO OBTENER EL DATASET?"
    echo "--------------------------------------------------------------------------------"
    echo ""
    echo "🔹 OPCIÓN 1: Si ya tienes DVC instalado en tu máquina:"
    echo "   Ejecuta en tu terminal local (PowerShell / CMD / Bash):"
    echo "   --------------------------------------------------------"
    echo "   dvc pull"
    echo "   --------------------------------------------------------"
    echo "   (En Windows con el entorno virtual del proyecto: .\\.venv-dvc\\Scripts\\dvc.exe pull)"
    echo ""
    echo "🔹 OPCIÓN 2: Si NO tienes DVC instalado en tu máquina:"
    echo "   1. Instala DVC con soporte para Google Drive:"
    echo "      pip install dvc dvc-gdrive"
    echo "   2. Descarga el dataset:"
    echo "      dvc pull"
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "🚀 PASO SIGUIENTE TRAS DESCARGAR EL ARCHIVO:"
    echo "--------------------------------------------------------------------------------"
    echo "Una vez que 'data/raw/salary.csv' esté presente en tu carpeta, ejecuta:"
    echo ""
    echo "   docker compose up"
    echo ""
    echo "================================================================================"
    echo "⛔ Cerrando ejecución del contenedor."
    echo "================================================================================"
    echo ""
    exit 1
fi

echo ""
echo "================================================================================"
echo "✅ DATASET ENCONTRADO: data/raw/salary.csv"
echo "🚀 INICIANDO PIPELINE DE MACHINE LEARNING (SPC PROJECT)"
echo "================================================================================"
echo ""

echo "📊 [1/3] Limpiando dataset (salary.csv -> data/interim)..."
python -m spc_module.dataset

echo ""
echo "⚙️ [2/3] Generando tabla minable y partición Train/Test (data/processed)..."
python -m spc_module.features

echo ""
echo "🤖 [3/3] Entrenando modelos (XGBoost y Red Neuronal MLP) y registrando en MLflow..."
python -m spc_module.modeling.train

echo ""
echo "================================================================================"
echo "🎉 PIPELINE COMPLETADO EXITOSAMENTE"
echo "🌐 Servidor MLflow UI iniciado y disponible en:"
echo "👉 http://localhost:8080"
echo "================================================================================"
echo ""

exec mlflow ui --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///mlflow.db
