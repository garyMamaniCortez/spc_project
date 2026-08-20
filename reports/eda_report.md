# Reporte EDA — salary.csv (Adult Census Income)

## 1. Resumen general
- Filas: **32561**
- Columnas: **15**
- Filas duplicadas: **24**
- Columnas con valores faltantes: **occupation, workclass, native-country**

## 2. Valores faltantes
|                |   missing_count |   missing_pct |
|:---------------|----------------:|--------------:|
| occupation     |            1843 |          5.66 |
| workclass      |            1836 |          5.64 |
| native-country |             583 |          1.79 |

## 3. Outliers (regla IQR)
| column         |   n_outliers |   pct_outliers |
|:---------------|-------------:|---------------:|
| hours-per-week |         9008 |          27.66 |
| capital-gain   |         2712 |           8.33 |
| capital-loss   |         1519 |           4.67 |
| education-num  |         1198 |           3.68 |
| fnlwgt         |          992 |           3.05 |
| age            |          143 |           0.44 |

## 4. Estadísticas numéricas
|                |   count |      mean |       std |   min |    25% |    50% |    75% |            max |   skew |   kurtosis |
|:---------------|--------:|----------:|----------:|------:|-------:|-------:|-------:|---------------:|-------:|-----------:|
| age            |   32561 |     38.58 |     13.64 |    17 |     28 |     37 |     48 |    90          |   0.56 |      -0.17 |
| fnlwgt         |   32561 | 189778    | 105550    | 12285 | 117827 | 178356 | 237051 |     1.4847e+06 |   1.45 |       6.22 |
| education-num  |   32561 |     10.08 |      2.57 |     1 |      9 |     10 |     12 |    16          |  -0.31 |       0.62 |
| capital-gain   |   32561 |   1077.65 |   7385.29 |     0 |      0 |      0 |      0 | 99999          |  11.95 |     154.8  |
| capital-loss   |   32561 |     87.3  |    402.96 |     0 |      0 |      0 |      0 |  4356          |   4.59 |      20.38 |
| hours-per-week |   32561 |     40.44 |     12.35 |     1 |     40 |     40 |     45 |    99          |   0.23 |       2.92 |

## 5. Figuras generadas
- `distributions`: 01_numerical_distributions.png
- `boxplots`: 02_boxplots_by_target.png
- `categorical`: 03_categorical_counts.png
- `correlation`: 04_correlation_heatmap.png
- `target_balance`: 05_target_balance.png