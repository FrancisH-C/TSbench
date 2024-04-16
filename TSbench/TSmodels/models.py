"""Model module defing BaseClass and subclasses."""

from __future__ import annotations

import json
import math
import os
import pickle
from abc import abstractmethod
from typing import Optional, Type, TypeVar, Union

import numpy as np
import pandas as pd
from numpy.random import PCG64, Generator

from TSbench.TSdata.data import AnyData, Data, size
from TSbench.TSdata.TSloader import LoaderTSdf, TSloader
from TSbench.TSmodels.point_process import Deterministic, PointProcess
from TSbench.TSmodels.utils.corr_mat import Corr_mat


class BaseModel:
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
        mat (np.ndarray, optional): A matrix to transform as Corr_mat.

    """

    loader: TSloader
    name: str
    dim: int
    lag: int
    rg: Generator
    corr_mat: Corr_mat
    point_process: PointProcess
    dim_label: np.ndarray
    feature_label: np.ndarray

    def __init__(
        self,
        loader: Optional[TSloader] = None,
        name: Optional[str] = None,
        dim: int = 1,
        lag: int = 1,
        rg: Optional[Generator] = None,
        corr_mat: Optional[np.ndarray] = None,
        point_process: Optional[PointProcess] = None,
        dim_label: Optional[np.ndarray] = None,
        feature_label: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize BaseModel."""
        self.dim = dim
        self.lag = lag

        if loader is None:
            self.loader = LoaderTSdf(datatype="Simulated")

        self.set_name(name)
        self.set_random_generator(rg)
        self.set_corr_mat(corr_mat)
        self.set_feature_label(feature_label)
        self.set_dim_label(dim_label=dim_label)
        self.set_point_process(point_process)

    def set_name(self, name: Optional[str] = None) -> None:
        self._name = name

    def set_point_process(self, point_process: Optional[PointProcess] = None) -> None:
        if point_process is None:
            point_process = Deterministic()
        self.point_process = point_process

    def set_random_generator(self, rg: Optional[Generator] = None) -> None:
        if rg is None:
            rg = Generator(PCG64())
        self.rg = rg

    def set_corr_mat(self, mat: Optional[np.ndarray] = None) -> None:
        if mat is None:
            self.corr_mat = Corr_mat(dim=self.dim, rg=self.rg)
        elif mat is not None:
            self.corr_mat = Corr_mat(mat=mat, rg=self.rg)

    def set_feature_label(self, feature_label: Optional[np.ndarray] = None) -> None:
        if feature_label is None:
            feature_label = np.array(["returns"])
        if len(feature_label) != 1:
            raise ValueError("Need 'feature_label' with 1 entry")
        self.feature_label = feature_label

    def set_dim_label(self, dim_label: Optional[np.ndarray] = None) -> None:
        if dim_label is None:
            dim_label = np.array(list(map(str, np.arange(self.dim))))
        elif len(dim_label) != self.dim:
            raise ValueError("Dimension mismatch between `dim_label` and `dim`")

        self.dim_label = dim_label

    def __str__(self) -> str:
        """Model basic string info."""
        if self._name is None:
            self._name = type(self).__name__.replace("_", "-")
        return self._name

    def __repr__(self) -> str:
        """Model representation."""
        return str(self)

    def get_timestamp(
        self,
        start: Optional[int] = None,
        start_index: Optional[int] = None,
        end: Optional[int | str] = None,
        end_index: Optional[int] = None,
    ) -> np.ndarray:
        return self.loader.get_timestamp(
            start=start,
            start_index=start_index,
            end=end,
            end_index=end_index,
            IDs=np.array([str(self)]),
        )

    def get_data(
        self,
        tstype: Type[Data] = pd.DataFrame,
        start: Optional[int | str] = None,
        start_index: Optional[int] = None,
        end: Optional[int | str] = None,
        end_index: Optional[int] = None,
        timestamps: Optional[slice | np.ndarray] = None,
        dims: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> Data:
        """Get the model's data.

        For format LoaderTSdf return the time series For format other
        than LoaderTSdf, returns only the observations of the time
        series.

        """
        return self.loader.get_timeseries(
            IDs=np.array([str(self)]),
            start=start,
            start_index=start_index,
            end=end,
            end_index=end_index,
            timestamps=timestamps,
            dims=dims,
            features=features,
            tstype=tstype,
        )

    def set_data(
        self,
        data: Optional[AnyData] = None,
        reset_timestamp: bool = True,
        collision: str = "overwrite",
    ) -> AnyData:
        """Set model's data using loader.

        For format LoaderTSdf set the timeseries. For data of type
        other than LoaderTSdf, it assumes that data represents
        observations of the timeseries.

        """
        if data is None or size(data) == 0:
            self.point_process.set_current_timestamp(current_timestamp=0)
            self.loader.add_data(data=None, ID=str(self), collision=collision)
            return self.get_data()
        if reset_timestamp:
            self.point_process.set_current_timestamp(current_timestamp=0)
        else:
            self.point_process.set_current_timestamp(
                current_timestamp=self.get_timestamp(start_index=-1)[0] + 1
            )
        timestamp = self.point_process.generate_timestamp(nb_points=size(data))

        self.loader.add_data(
            data=data,
            ID=str(self),
            timestamp=timestamp,
            collision=collision,
            dim_label=self.dim_label,
            feature_label=self.feature_label,
        )
        return self.get_data()

    def register_data(
        self,
        loader: TSloader,
        ID: Optional[str] = None,
        append_to_feature: Optional[str] = None,
        feature_label: Optional[np.ndarray] = None,
        collision="update",
    ) -> AnyData:
        """Record outputs from model to an external loader.

        Record data under given ID (used for data forecast).
        If None, it records under its own name (used for data generated)

        model.data (list[np.ndarray]): Every list entry (of type
            np.ndarray) is a feature with dimension T \times self.dim.
            The list is length is the length of the features.

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
            feature_label = np.array(
                [
                    self.feature_label[i] + "_" + append_to_feature
                    for i in range(len(self.feature_label))
                ]
            )

        if data.shape[1] != len(feature_label):
            raise ValueError("Need the same length of data as the lenght of feature .")

        loader.add_data(
            data=data, ID=ID, feature_label=feature_label, collision=collision
        )
        return loader.df

    def save_model(self, filename: str) -> None:
        """Save model.

        If to be reuse in Python, "pickle" is prefered.

        Caution: With json there is loss of information.

        Args:
            filename (str): Name of pickle file where to save.
        """

        def convert_to_json_serializable(model: TSloader) -> dict:
            attributes = model.__dict__
            to_output = {}
            for att_name in attributes:
                if type(attributes[att_name]) is np.ndarray:
                    to_output[att_name] = attributes[att_name].tolist()
                if isinstance(attributes[att_name], Union[int, float, complex]):
                    to_output[att_name] = attributes[att_name]
            return to_output

        _, ext = os.path.splitext(filename)
        if ext == ".pkl" or ext == ".pickle":
            with open(filename, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif ext == ".json":
            with open(filename, "w") as f:
                f.write(
                    json.dumps(
                        self,
                        default=convert_to_json_serializable,
                        sort_keys=True,
                        indent=4,
                    )
                )
        else:
            raise ValueError("Unsupported extension type.")

    SubModel = TypeVar("SubModel", bound="BaseModel")

    @classmethod
    def load_model(cls: Type[SubModel], filename: str) -> SubModel:
        """Load model.

        "pickle" is prefered to load the exact model.

        Caution: json does not differentiate between list and array, models may.

        Args:
            filename (str): Name of pickle file where to save.
        """
        SubModel = TypeVar("SubModel", bound="BaseModel")

        def model_from_parameters(
            cls, parameters: dict[str, Union[int, float, complex, list]]
        ) -> SubModel:
            "returns a new instance of the class"
            new_instance = cls.__new__(cls)
            new_instance.__init__(**parameters)
            return new_instance

        _, ext = os.path.splitext(filename)
        if ext == ".pkl" or ext == ".pickle":
            with open(filename, "rb") as f:
                return pickle.load(f)
        elif ext == ".json":
            with open(filename, "r") as f:
                return model_from_parameters(cls, json.load(f))
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
        self, N: int, reset_timestamp: bool = True, collision: str = "overwrite"
    ) -> AnyData:
        """Generate `T` values using Model.

        Args:
            T (int): Number of observations to generate.

        Returns:
            np.ndarray: data
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
    def train(self) -> "BaseModel":
        """Train model using `state` as the trainning set.

        Args:
            data (np.ndarray): Input data.

        Returns:
            Model: Trained model.

        """
        pass

    @abstractmethod
    def forecast(
        self,
        T: int,
        reset_timestamp: bool = False,
        collision: str = "overwrite",
    ) -> AnyData:
        """Forecast a timeseries.

        Knowing `series` from `start_index`, set self.timeseries to `T`
        forecast values.

        Args:
            data (np.ndarray): Input data.
            T (int): Number of forward forecast.

        Returns:
            dict[str, np.ndarray]: Possible {key :value} outputs
                - {"returns" : .ndarray of returns}
                - {"vol" : np.ndarray of vol}.
        """
        pass

    def rolling_forecast(
        self,
        T: int,
        batch_size: int = 1,
        window_size: int = 0,
        train: bool = False,
        side: str = "before",
    ) -> AnyData:
        """Rolling forecast a timeseries.

        rounding (default = "before") : Control the behavior when
            T/batch_size != 0, does it stop 'before' T (default) or
            'after' T.

        window_size (default = 0) : How much data to preserve. By
            default 0 to keep all data, i.e. doing an expanding window
            forecast.

        """
        # How to round forecast
        if side == "before":
            T = math.floor(T / batch_size) * batch_size

        # start at last forecast timestamp
        rolling_forecast = pd.DataFrame()
        for _ in range(0, T, batch_size):
            # input
            self.set_data(self.get_data(start_index=-window_size))
            if train:
                self.train()
            self.forecast(batch_size, reset_timestamp=False, collision="update")

            # output
            forecast = self.get_data(start_index=-batch_size)
            rolling_forecast = pd.concat([rolling_forecast, forecast])
        self.set_data(rolling_forecast)
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
