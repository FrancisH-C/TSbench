"""Constant model."""
from __future__ import annotations
from TSbench.TSmodels.models import Model
import numpy as np
import pandas as pd


class Constant(Model):
    """A simple constant model."""

    def __init__(self, C: float = None, **model_args) -> None:
        """Initialize Constant."""
        super().__init__(**model_args)
        if C is None:
            C = self.rg.uniform(low=0, high=1, size=self.dim)
        self.C = C

    def generate(self, T: int) -> dict[str, np.array]:
        """Set self.outputs to `T` generated values using ARMA.

        Args:
            T (int): Number of observations to generate.

        Returns:
            dict[str, np.array]: {key :value} outputs
                - {"returns"  : np.array of returns}

        """
        self.outputs["returns"] = self.C * np.ones((T, self.dim))
        return self.outputs

    def train(self, series: pd.DataFrame) -> "Constant":
        """Train model using `series` as the trainning set.

        Args:
            series (pd.DataFrame): Input series.

        Returns:
            Constant: Trained Constant model.

        """
        self.C = np.mean(series.to_numpy())
        return self

    def forecast(
        self, series: pd.DataFrame, start_index: int, T: int
    ) -> dict[str, np.array]:
        """Forecast a timeseries.

        Knowing `series` from `start_index`, set self.forecasted to `T`
        forecasted values.

        Args:
            series (pd.DataFrame): Input series.
            start_index (int): Index corresponding to the series
                starting point in a rolling forecast. E.g. With a 100 rate
                rolling window. `start_index` will increment by a 100 on
                every "roll".
            T (int): Number of forward forecast.

        Returns:
            dict[str, np.array]: Possible {key :value} outputs
                - {"returns" : np.array of returns}
                - {"vol" : np.array of vol}.
        """
        self.forecasted = self.C * np.ones((T, self.dim))
        return self.forecasted
