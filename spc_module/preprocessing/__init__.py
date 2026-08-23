"""Preprocessing package: turns the raw salary dataset into a mineable table.

Responsibilities are split into small, single-purpose modules (SRP):

- :mod:`spc_module.preprocessing.cleaning` — column dropping, missing
  value imputation and duplicate removal.
- :mod:`spc_module.preprocessing.encoding` — categorical one-hot
  encoding and target binarization.
- :mod:`spc_module.preprocessing.scaling` — standardization of the
  continuous numeric columns only (never the one-hot dummies).
- :mod:`spc_module.preprocessing.splitting` — train/test split.
- :mod:`spc_module.preprocessing.builder` — orchestrates the above via
  dependency injection to produce the final mineable table.
"""
