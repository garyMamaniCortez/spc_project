"""Exploratory Data Analysis (EDA) sub-package for the salary project.

This package follows SOLID principles: each class has a single
responsibility (loading, quality checks, statistical profiling,
visualization) and the high-level orchestrator (:class:`EDAReport`)
depends on abstractions rather than concrete implementations.
"""

from spc_module.eda.loader import CSVDataLoader, DataLoader
from spc_module.eda.profiler import DataProfiler
from spc_module.eda.quality import DataQualityChecker
from spc_module.eda.report import EDAReport
from spc_module.eda.visualizer import BaseVisualizer, MatplotlibVisualizer

__all__ = [
    "BaseVisualizer",
    "CSVDataLoader",
    "DataLoader",
    "DataProfiler",
    "DataQualityChecker",
    "EDAReport",
    "MatplotlibVisualizer",
]
