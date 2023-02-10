"""Model class and implemented timeseries models."""
from TSbench.models.models import GeneratorModel, ForecastingModel, Model

from TSbench.models.simple import Constant

from TSbench.models.ARMA import ARMA

from TSbench.models.GARCH import GARCH


try:
    from TSbench.models.R.rGARCH import rGARCH
except ImportError:
    pass
