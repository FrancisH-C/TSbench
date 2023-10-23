"""Model class and implemented timeseries models."""

import importlib.util

from TSbench.TSmodels.models import GeneratorModel, ForecastingModel, Model

from TSbench.TSmodels.simple import Constant

from TSbench.TSmodels.ARMA import ARMA

from TSbench.TSmodels.GARCH import GARCH, VEC_GARCH, SPD_VEC_GARCH

if importlib.util.find_spec("rpy2") is not None:
    from TSbench.TSmodels.R import rGARCH

from TSbench.TSmodels.point_process import PointProcess, Deterministic

