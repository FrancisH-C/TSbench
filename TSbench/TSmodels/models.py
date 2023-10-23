"""Model module defing BaseClass and subclasses."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union
import inspect

from TSbench.TSdata.TSloader import TSloader, LoaderTSdf
from TSbench.TSdata.DataFormat import convert_to_TSdf
from TSbench.TSmodels.data import Data
from TSbench.TSmodels.utils.corr_mat import Corr_mat
from TSbench.TSmodels.point_process import Deterministic

import numpy as np
import pandas as pd
import math
from numpy.random import Generator
from randomgen import Xoshiro256

import os
import pickle
import json


class BaseModel(ABC):
    """Abstract Base model class.

    For model generating outputs or forecasting timeseries.
    Each `Basemodel` subclass needs to evaluate what to do when its
    parameters are undefined. They can have default, infer it using
    other model specific parameters or raise an error.

    Args:
        dim (int): Dimension of the model. Default is 1.
        lag (int, optional): How much lag should the model use.
        corr_mat (Corr_at, optional): A way to manipulate correlation
            matrices for the model. See Corr_mat for more info for how to
            se this parameter. Default is an identity correlation matrix.
        mat (np.array, optional): A matrix to transform as Corr_mat.

    """

    def __init__(
        self,
        loader = None,
        name = None,
        dim: int = 1,
        lag: int = None,
        rg: Generator = None,
        corr_mat: Corr_mat = None,
        point_process = None,
        dim_label: list[str] = None,
        feature_label: list[str] = None,
        default_features: list[str] = ["returns"]
    ) -> None:
        """Initialize BaseModel."""
        self.dim = dim
        self.lag = lag
        self.default_features = default_features

        if loader is None:
            self.loader = LoaderTSdf(datatype="Simulated")

        self.set_name(name)
        self.set_random_generator(rg)
        self.set_corr_mat(corr_mat)
        self.set_feature_label(feature_label)
        self.set_dim_label(dim_label= self.set_point_process())

    def set_name(self, name=None):
        self._name = name

    def set_point_process(self, point_process=None):
        if point_process is None:
            point_process = Deterministic()
        self.point_process = point_process

    def set_random_generator(self, rg=None):
        if rg is None:
            rg = Generator(Xoshiro256())
        self.rg = rg

    def set_corr_mat(self, corr_mat=None):
        if corr_mat is None:
            self.corr_mat = Corr_mat(dim=self.dim, rg=self.rg, method="uncorrelated")
        elif corr_mat is not None:
            self.corr_mat = Corr_mat(mat=corr_mat, rg=self.rg)
        else:
            corr_mat.dim = self.dim
            corr_mat.set_mat()
            self.corr_mat = corr_mat

    def default_feature_label(self):
        if self.dim > 1:
            return [self.default_features[i] + str(j) for i in range(len(self.default_features)) for j in range(self.dim)]
        else:
            return [self.default_features[i] for i in range(len(self.default_features))]

        

        return [self.default_features[i] + str(j) for i in range(len(self.default_features)) for j in range(self.dim)]

    def set_feature_label(self, feature_label=None):
        if feature_label is None:
            feature_label = self.default_feature_label()
        #elif len(feature_label) != len(self.default_features) * self.dim:
        #    raise ValueError("Need `nb_features` entry(ies) for 'feature_label'")
        self.feature_label = feature_label

    def set_dim_label(self, dim_label=None):
        if dim_label is None:
            dim_label = list(map(str, np.arange(self.dim)))
        elif len(dim_label) != self.dim:
            raise ValueError("Dimension mismatch between `dim_label` and `dim`")

        self.dim_label = dim_label

    def __str__(self) -> str:
        """Model basic string info."""
        if self._name == None:
            self._name = type(self).__name__.replace("_", "-") 
        return self._name

    def __repr__(self) -> str:
        """Model representation."""
        return str(self)

    def get_data(self, format: type = LoaderTSdf, start = None, start_index = None, end = None, end_index = None, timestamps = None, dims = None, features = None):
        """Get the model's data.

        For format LoaderTSdf return the time series
        For format other than LoaderTSdf, returns only the observations of the time series.
        """
        data = self.loader.get_timeseries(IDs=[str(self)], start=start, start_index=start_index, end=end, end_index=end_index, timestamps=timestamps, dims=dims, features=features)
        if format is LoaderTSdf:
            return data
        if format is np.ndarray:
            return data.to_numpy()
        if format is pd.DataFrame:
            return data

    def set_data(self, data: Data = None, timestamp: np.array=None, reset_timestamp=True, collision: str ="overwrite"):
        """Set data using loader.

        For format LoaderTSdf set the timeseries.
        For data of type other than LoaderTSdf, it assumes that data represents observations of the timeseries.

        """
        if data is None or len(data) == 0:
            self.point_process.set_current_timestamp(current_timestamp=0)
            self.loader.add_data(data=None, ID=str(self), collision = collision)
            return self.get_data()

        if timestamp is None:
            if reset_timestamp:
                self.point_process.set_current_timestamp(current_timestamp=0)
            timestamp = self.point_process.generate_timestamp(nb_points=len(data))
        self.point_process.set_current_timestamp(current_timestamp=timestamp[-1] + 1)

        self.loader.add_data(data=data, ID=str(self), timestamp=timestamp, collision = collision)
        return self.get_data()

    def register_data(self, loader=None, ID=None, append_to_feature = None, feature_label=None, collision="update"):
        """Record outputs.

        Record data under given ID (used for data forecaste).
        If None, it records under its own name (used for data generated)

        model.data (list[np.array]): Every list entry (of type np.array) is a feature
            with dimension T \times self.dim. The list is length is the length of the features.
            i.e. 
                    len(data) == len(feature).
                    [len(data[i]) == T for i in range(len(data))]
        """
        ## record with ID and feature_label from the model (used for generated data)
        data = self.get_data()
        if ID is None:
            ID = str(self)

        if append_to_feature is None:
            feature_label = self.feature_label
        else:
            feature_label = [self.feature_label[i] + "_" + append_to_feature for i in range(len(self.feature_label))]

        if data.shape[1] != len(feature_label):
            raise ValueError("Need the same length of data as the lenght of feature .")

        loader.add_data(
            data=data,
            ID = ID,
            feature_label=feature_label,
            collision=collision
        )
        return loader.df

    def save_model(self, filename: str) -> None:
        """Save model.

        If to be reuse in Python, "pickle" is prefered.

        Caution: With json there is loss of information.

        Args:
            filename (str): Name of pickle file where to save.
        """
        def convert_to_json_serializable(model):
            attributes = model.__dict__
            to_output = {}
            for att_name in attributes:
                if type(attributes[att_name]) is np.ndarray:
                    to_output[att_name] = attributes[att_name].tolist()
                if isinstance(attributes[att_name],  Union[int, float, complex]):
                    to_output[att_name] = attributes[att_name]
            return to_output

        _, ext = os.path.splitext(filename)
        if ext == ".pkl" or ext == ".pickle":
            with open(filename, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif ext == ".json":
            with open(filename, 'w') as f:
                f.write(json.dumps(self, default=convert_to_json_serializable, sort_keys=True, indent=4))
        else:
            raise ValueError("Unsupported extension type.")

    @classmethod
    def load_model(model_class, filename: str) -> None:
        """Load model.

        "pickle" is prefered to load the exact model.
 
        Caution: json does not differentiate between list and array, models may.

        Args:
            filename (str): Name of pickle file where to save.
        """
        def model_from_parameters(model_class, parameters: dict[str, Union[int, float, complex, list]]):
            "returns a new instance of the class"
            new_instance = model_class.__new__(model_class)
            new_instance.__init__(**parameters)
            return new_instance
        
        _, ext = os.path.splitext(filename)
        if ext == ".pkl" or ext == ".pickle":
            with open(filename, 'rb') as f:
                return pickle.load(f)
        elif ext == ".json":
            with open(filename, 'r') as f:
                return model_from_parameters(model_class, json.load(f))
        else:
            raise ValueError("Unsupported extension type.")


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

    def init_length(self) -> int:
        """Length to initialize `self.timeseries`."""
        return self.lag

    @abstractmethod
    def generate(
        self, N: int, reset_timestamp=True, collision: str = "overwrite"
    ) -> Model:
        """Generate `T` values using Model.

        Args:
            T (int): Number of observations to generate.

        Returns:
            np.array: data
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

    @abstractmethod
    def train(self, data: Data = None) -> "Model":
        """Train model using `state` as the trainning set.

        Args:
            data (np.array): Input data.

        Returns:
            Model: Trained model.

        """
        pass

    @abstractmethod
    def forecast(
        self,
        T: int,
        reset_timestamp=True,
        collision: str = "overwrite",
    ) -> Data:
        """Forecast a timeseries.

        Knowing `series` from `start_index`, set self.timeseries to `T`
        forecast values.

        Args:
            data (np.array): Input data.
            T (int): Number of forward forecast.

        Returns:
            dict[str, np.ndarray]: Possible {key :value} outputs
                - {"returns" : .ndarray of returns}
                - {"vol" : np.array of vol}.
        """
        pass

    def rolling_forecast(
            self, T: int, batch_size=1, window_size=0,
            train: bool = False,
            side="before", collision: str ="overwrite"
    ) -> Data:
        """Rolling forecast a timeseries.

        rounding (default = "before") : Control the behavior when T/batch_size != 0,
            does it stop 'before' T (default) or 'after' T.

        window_size (default = 0) : How much data to preserve. By default 0 to keep all data,
            i.e. doing an expanding window forecast.
        """
        # How to round forecast
        if side == "before":
            T = math.floor(T / batch_size) * batch_size

        if T >= batch_size:
            # first forecast overwriting observations
            x = self.get_data(start_index=-window_size)
            print(x)
            self.set_data(self.get_data(start_index=-window_size))
            self.forecast(batch_size, collision="overwrite")

        print("you see me rolling")
        # start at last forecast timestamp
        for _ in range(batch_size, T, batch_size):
            x = self.get_data(start_index=-window_size)
            print(x)
            self.set_data(self.get_data(start_index=-window_size))
            self.forecast(batch_size, collision="update")
        return self.get_data()


class Model(GeneratorModel, ForecastingModel):
    """Model class.

    For model generating outputs and forecasting data.

    Args:
        **model_args: Arguments for `models` `Model`.
            Possible keywords are `dim`, `lag` and `corr_mat`.
    """

    def __init__(self, **model_args) -> None:
        """Initialize Model."""
        super().__init__(**model_args)

    def generate_mode():
        self.generate_mode = True
        self.forecast_mode = False

    def forecast_mode():
        self.forecast_mode = True
        self.generate_mode = False
