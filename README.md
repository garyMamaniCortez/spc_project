# Salary Prediction Clasification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<br>
## 👥 Integrantes — Grupo 3

- Diego Alvarado García
- Erick Alejandro Quiroz Gil
- Sebastian Gustavo Marín Ovando
- Ayelen Ortiz Robledo
- Jhonny Gary Mamani Cortez


Salary Prediction Clasification project

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         spc_module and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── spc_module   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes spc_module a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Limpia el raw salary.csv (drop columnas,
    │                              imputación por moda, duplicados) -> data/interim
    │
    ├── features.py             <- Construye la tabla minable: target binario +
    │                              one-hot encoding + split train/test -> data/processed
    │
    ├── preprocessing/          <- Tabla minable (SOLID): pasos reutilizables
    │   ├── __init__.py
    │   ├── cleaning.py         <- CleaningStep (Strategy) + CleaningPipeline
    │   ├── encoding.py         <- OneHotCategoricalEncoder + BinaryTargetEncoder
    │   ├── splitting.py        <- DatasetSplitter (train/test estratificado)
    │   └── builder.py          <- MineableTableBuilder (orquestador, DIP)
    │
    ├── eda/                    <- Análisis exploratorio (loader, quality, profiler...)
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Tabla minable (dataset `salary.csv`, Adult Census Income)

La tarea es de **clasificación binaria**: predecir si una persona gana
`>50K` o `<=50K` al año. El pipeline para generar la tabla minable
sigue el patrón cookiecutter (`raw` → `interim` → `processed`) y
aplica principios SOLID (cada paso es una clase de responsabilidad
única, inyectada por dependencia en `MineableTableBuilder`):

```bash
python spc_module/dataset.py      # raw/salary.csv -> interim/salary_clean.csv
python spc_module/features.py     # interim/salary_clean.csv -> processed/{features,labels,test_features,test_labels}.csv
```

Decisiones de diseño aplicadas:

- **`fnlwgt`** se elimina: es un peso muestral censal, no una variable predictiva.
- **`education`** se elimina: es redundante con `education-num` (ya numérica/ordinal).
- **Valores `"?"`** en `workclass`, `occupation` y `native-country` se imputan con la **moda** de cada columna.
- **`salary`** se binariza a `0`/`1` (`1` = `>50K`).
- Variables categóricas restantes se codifican con **one-hot encoding** (`drop_first=True` por defecto, para evitar la trampa de la variable ficticia en modelos lineales; configurable vía CLI).
- Split **train/test estratificado** (80/20 por defecto) para preservar el desbalance de clases (~75% / 25%).

--------

