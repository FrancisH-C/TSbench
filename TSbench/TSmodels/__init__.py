"""BaseModel's subclasses and implemented timeseries models."""

import importlib.util

from TSbench.TSmodels.ARMA import ARMA
from TSbench.TSmodels.GARCH import GARCH, VEC_SPD_GARCH, VEC_GARCH
from TSbench.TSmodels.models import ForecastingModel, GeneratorModel, Model
from TSbench.TSmodels.simple import Constant

if importlib.util.find_spec("rpy2") is not None:
    from TSbench.TSmodels.R import rGARCH

from TSbench.TSmodels.point_process import PointProcess, Deterministic

__all__ = [
    "ARMA",
    "GARCH",
    "VEC_GARCH",
    "VEC_SPD_GARCH",
    "ForecastingModel",
    "GeneratorModel",
    "Model",
    "rGARCH",
    "Constant",
    "Deterministic",
    "PointProcess",
]
