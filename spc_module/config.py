from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Salary dataset (Adult Census Income) specific configuration ---
TARGET_COLUMN = "salary"
POSITIVE_LABEL = ">50K"

# fnlwgt: census sampling weight, not a predictive feature.
# education: redundant with the already numeric/ordinal education-num.
COLUMNS_TO_DROP = ["fnlwgt", "education"]

# Columns whose raw "?" values (parsed as NaN by CSVDataLoader) are
# imputed with their mode.
CATEGORICAL_COLUMNS_TO_IMPUTE = ["workclass", "occupation", "native-country"]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
