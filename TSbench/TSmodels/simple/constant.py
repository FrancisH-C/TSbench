"""Constant model."""

from __future__ import annotations

import numpy as np

from TSbench.TSdata.data import AnyData
from TSbench.TSmodels.models import Model
from typing import Optional


class Constant(Model):
    """A simple constant model."""

    def __init__(self, constant: Optional[np.ndarray] = None, **model_args) -> None:
        """Initialize Constant."""
        super().__init__(**model_args)
        if constant is None:
            constant = self.rg.uniform(low=0, high=1, size=self.dim)
        self.constant = constant

    def generate(
        self, N: int, reset_timestamp: bool = True, collision: str = "overwrite"
    ) -> AnyData:
        """Generate `T` values using Constant.

        Args:
            T (int): Number of observations to generate.

        Returns:
            np.ndarray: returns

        """
        return self.set_data(
            data=self.constant * np.ones((N, self.dim)),
            reset_timestamp=reset_timestamp,
            collision=collision,
        )

    def train(self) -> "Constant":
        """Train model using `data` as the trainning set.

        Args:
            data (np.ndarray): Input data.

        Returns:
            Constant: Trained Constant model.

        """
        self.constant = np.nanmean(self.get_data())
        return self

    def forecast(
        self,
        T: int,
        reset_timestamp: bool = False,
        collision: str = "overwrite",
    ) -> AnyData:
        """Forecast a data.

        Knowing `data` forecast `T` values.

        Args:
            data (np.ndarray): Input data.
            T (int): Number of forward forecast.

        Returns:
            dict[str, np.ndarray]: Possible {key :value} outputs
                - {"returns" : np.ndarray of returns}
        """
        return self.set_data(
            data=self.constant * np.ones((T, self.dim)),
            reset_timestamp=reset_timestamp,
            collision=collision,
        )
