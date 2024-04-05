"""Import GARCH functions."""

from TSbench.TSmodels.GARCH.GARCH import GARCH
from TSbench.TSmodels.GARCH.MGARCH import VEC_SPD_GARCH, VEC_GARCH

__all__ = ["GARCH", "VEC_SPD_GARCH", "VEC_GARCH"]
