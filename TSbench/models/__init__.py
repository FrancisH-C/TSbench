"""Model class and implemented timeseries models."""
from TSbench.models.models import GeneratorModel, ForecastingModel, Model

from TSbench.models.ARMA import ARMA

try:
    from TSbench.models.R.rGARCH import rGARCH
except ImportError:
    pass
