"""Model module defing BaseClass and subclasses."""
from __future__ import annotations
from abc import ABC, abstractmethod

from TSbench.TSmodels.utils.corr_mat import Corr_mat

import pandas as pd
import numpy as np
from numpy.random import Generator
from randomgen import Xoshiro256

import pickle
import json


class BaseModel(ABC):
    """Base model class.

    For model generating outputs or forecasting timeseries.
    Each `Basemodel` subclass needs to evaluate what to do when its
    parameters are undefined. They can have default, infer it using
    other model specific parameters or raise an error.

    Args:
        dim (int): Dimension of the model. Default is 1.
        lag (int, optional): How much lag should the model use.
        corr_mat (Corr_mat, optional): A way to manipulate correlation
            matrices for the model. See Corr_mat for more info for how to
            se this parameter. Default is an identity correlation matrix.
        mat (np.array, optional): A matrix to transform as Corr_mat.
    """

    def __init__(
        self,
        dim: int = 1,
        lag: int = None,
        features: list[str] = ["returns"],
        rg: Generator = None,
        corr_mat: Corr_mat = None,
    ) -> None:
        """Initialize BaseModel."""
        self.dim = dim
        self.lag = lag

        if rg is None:
            rg = Generator(Xoshiro256())
        self.rg = rg

        if corr_mat is None:
            self.corr_mat = Corr_mat(dim=dim, rg=self.rg, method="uncorrelated")
        elif corr_mat is not None:
            self.corr_mat = Corr_mat(mat=corr_mat, rg=self.rg)
        else:
            corr_mat.dim = self.dim
            corr_mat.set_mat()
            self.corr_mat = corr_mat

    def __str__(self) -> str:
        """Model basic string info."""
        return type(self).__name__.replace("_", "-")

    def __repr__(self) -> str:
        """Model representation."""
        return type(self).__name__.replace("_", "-")

    def save_model(self, filename: str, ext: str = "pkl") -> None:
        """Save model.

        If to be reuse in Python, "pickle" is prefered.

        Otherwise, uses json

        Args:
            filename (str): Name of pickle file where to save.
        """
        if ext == "pkl":
            pickle.dump(self, filename)
        if ext == "pkl":
            json.dump(self, filename)


class GeneratorModel(BaseModel):
    """Model use to generate timeseries.

    For model generating outputs.

    Args:
        **model_args: Arguments for `models` `Model`.
            Possible keywords are `dim`, `lag` and `corr_mat`.

    """

    def __init__(self, **model_args) -> None:
        """Initialize GeneratorModel."""
        super().__init__(**model_args)
        self.outputs = {}

    def record_outputs(self, **outputs: np.array):
        """Record outputs."""
        for series_name in outputs:
            self.outputs[series_name] = outputs[series_name]

    def init_length(self) -> int:
        """Length to initialize `self.outputs`."""
        return self.lag

    @abstractmethod
    def generate(self, series: pd.DataFrame) -> "Model":
        """Train model using `series` as the trainning set.

        Args:
            series (pd.DataFrame): Input series.

        Returns:
            Model: Trained model.

        """
        pass


class ForecastingModel(BaseModel):
    """Model use to generate timeseries.

    For all model forecasting timeseries.

    Args:
        **model_args: Arguments for `models` `Model`.
            Possible keywords are `dim`, `lag` and `corr_mat`.

    """

    def __init__(self, **model_args) -> None:
        """Initialize ForecastingModel."""
        super().__init__(**model_args)
        self.forecasted = {}

    @abstractmethod
    def train(self, series: pd.DataFrame) -> "Model":
        """Train model using `series` as the trainning set.

        Args:
            series (pd.DataFrame): Input series.

        Returns:
            Model: Trained model.

        """
        pass

    @abstractmethod
    def forecast(
        self, serie: pd.DataFrame, start_index: int, T: int, retrain: bool = False
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
            dict[str, np.ndarray]: Possible {key :value} outputs
                - {"returns" : .ndarray of returns}
                - {"vol" : np.array of vol}.
        """
        pass


class Model(GeneratorModel, ForecastingModel):
    """Model class.

    For model generating outputs and forecasting timeseries.

    Args:
        **model_args: Arguments for `models` `Model`.
            Possible keywords are `dim`, `lag` and `corr_mat`.
    """

    def __init__(self, **model_args) -> None:
        """Initialize Model."""
        super().__init__(**model_args)
