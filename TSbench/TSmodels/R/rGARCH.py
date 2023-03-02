"""Univariate Garch model using R."""
# https://medium.com/analytics-vidhya/calling-r-from-python-magic-of-rpy2-d8cbbf991571
# https://medium.com/@remycanario17/update-converting-python-dataframes-to-r-with-rpy2-59edaef63e0e

import os
import numpy as np
import pandas as pd

import rpy2
from rpy2 import robjects as ro

from TSbench.TSmodels import ForecastingModel
from TSbench.TSmodels.R.Rpath import Rmodels_path

# Defining the R script and loading the instance in Python
r = rpy2.robjects.r
r["source"](os.path.normpath(os.path.join(Rmodels_path, "rGARCH.R")))


class rGARCH(ForecastingModel):
    """Use rugarch for GARCH model."""

    def __init__(self, **model_args) -> None:
        """Initialize rGARCH."""
        self.lag = 1
        super().__init__(**model_args)

    def train(self, series: pd.DataFrame) -> "rGARCH":
        """Train model using `series` as the trainning set.

        Args:
            series (pd.DataFrame): Input series.

        Returns:
            rGARCH: Trained rGARCH model.

        """
        # converting it into r object for passing into r function
        series_r = ro.vectors.FloatVector(series.to_numpy())

        # Loading the function we have defined in R.
        # Careful with naming since the global environment is used
        # !!! Don't use the same name for functions in R
        train_GARCH_r = rpy2.robjects.globalenv["train_ruGARCH"]
        # Invoking the R function and getting the result
        train_GARCH_r(series_r, self.lag, self.lag)
        return self

    def forecast(
        self, serie: pd.DataFrame, start_index: int, T: int, retrain: bool = False
    ) -> dict[str, np.array]:
        pass
