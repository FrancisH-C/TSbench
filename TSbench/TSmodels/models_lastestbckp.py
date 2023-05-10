"""Model module defing BaseClass and subclasses."""
from __future__ import annotations
from math import ceil
import numpy as np
from TSbench.utils.corr_mat import Corr_mat
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from abc import ABC, abstractmethod
from numpy.random import Generator
from randomgen import Xoshiro256


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
        rg: Generator = None,
        corr_mat: Corr_mat = None,
        mat: np.array = None,
    ) -> None:
        """Initialize BaseModel."""
        self.dim = dim
        self.lag = lag

        if rg is None :
            rg = Generator(Xoshiro256())
        self.rg = rg

        if corr_mat is None:
            self.corr_mat = Corr_mat(dim=dim, rg=self.rg, method="uncorrelated")
        elif mat is not None:
            self.corr_mat = Corr_mat(mat=mat, rg=self.rg)
        else:  # it is a corr_mat
            corr_mat.dim = self.dim
            corr_mat.set_mat()
            self.corr_mat = corr_mat

    def __str__(self) -> str:
        """Model basic string info."""
        return type(self).__name__.replace("_", "-")

    def __repr__(self) -> str:
        """Model representation."""
        return type(self).__name__.replace("_", "-")

    def save(self, filename: str) -> None:
        """Save model as pickle.

        Args:
            filename (str): Name of pickle file where to save.
        """
        pickle.dump(self, filename)


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
    def generate(self, T: int) -> dict[str, np.array]:
        """Set self.outputs to `T` generated values.

        Args:
            T (int): Number of observations to generate.

        Returns:
            dict[str, np.array]: Two standard {key :value} outputs
                - {"returns"  : np.array of returns}
                - {"vol"  : np.array of vol}.

        """
        pass

    def get_outputs(
        self,
        get_the: list = None,
        dtype: type = None,
        id: str = None,
        dates: pd.DatetimeIndex = None,
        dimensions: int = None,
    ) -> dict[str, np.array] | list[float] | np.array | pd.DataFrame:
        """Get subset of outputs from the model in a given format.

        Args:
            get_the (list[str], optional): The name of the outputs you want.
                If "all" is given, returns all of the outputs. It depends on
                the model but two standard `keys` are : {"returns","vol"}.
                Default is to output "all".
            dtype (str, optional): {dict`, list, np.array, pd.DataFrame}
                The way outputs is return. Default is `dict`.
            id (str, optional): Entity id to return.
            dates (pd.DatetimeIndex, optional): Dates to set as index.
            dimensions (int, optional): Dimensions to return.

        Returns:
            dict[str, np.array] | np.array,list[float] | pd.DataFrame:
                The data in the `dtype` format.
        """
        # what to return
        if get_the is None or get_the == "all":
            get_the = self.outputs.keys()
        elif type(get_the) is str:
            get_the = [get_the]

        # default : return dict
        if dtype is None:
            dtype = dict

        # how to return
        if dtype is dict:
            output = {}
            for get in get_the:
                output[get] = self.outputs[get]
            return output
        elif dtype is np.array:
            return self.outputs_to_array(get_the, dtype, id, dates, dimensions)
        elif dtype is list:
            return self.outputs_to_list(get_the, dtype, id, dates, dimensions)
        # else
        raise ValueError("`dtype` is either dict, list, np.array")

    def outputs_to_list(
        self,
        get_the: list = None,
        dtype: type = None,
        id: str = None,
        dates: pd.DatetimeIndex = None,
        dimensions: int = None,
    ) -> list[float]:
        """Get subset of outputs from the model in list.

        Args:
            get_the (list[str], optional): The name of the outputs you want.
                If "all" is given, returns all of the outputs. It depends on
                the model but two standard `keys` are : {"returns","vol"}.
                Default is to output "all".
            dtype (str, optional): {dict`, list, np.array, pd.DataFrame}
                The way outputs is return. Default is `dict`.
            id (str, optional): Entity id to return.
            dates (pd.DatetimeIndex, optional): Dates to set as index.
            dimensions (int, optional): Dimensions to return.

        Returns:
            dict[str, np.array] | np.array | list[float] | pd.DataFrame:
                The data in the `dtype` format.
        """
        output = []
        for get in get_the:
            output.append(self.outputs[get].tolist())
        return output

    def outputs_to_array(
        self,
        get_the: list = None,
        dtype: type = None,
        id: str = None,
        dates: pd.DatetimeIndex = None,
        dimensions: int = None,
    ) -> np.array:
        """Get subset of outputs from the model in numpy array.

        Args:
            get_the (list[str], optional): The name of the outputs you want.
                If "all" is given, returns all of the outputs. It depends on
                the model but two standard `keys` are : {"returns","vol"}.
                Default is to output "all".
            dtype (str, optional): {dict`, list, np.array, pd.DataFrame}
                The way outputs is return. Default is `dict`.
            id (str, optional): Entity id to return.
            dates (pd.DatetimeIndex, optional): Dates to set as index.
            dimensions (int, optional): Dimensions to return.

        Returns:
            dict[str, np.array] | np.array | list[float] | pd.DataFrame:
                The data in the `dtype` format.
        """
        T = list(self.outputs.values())[0].shape[0]

        output = np.zeros((T, self.dim * len(get_the)))
        k = 0
        for get in get_the:
            output[:, k : k + self.dim] = self.outputs[get]
        k += self.dim
        output = np.zeros(T)
        for get in get_the:
            output.append(self.outputs[get])
        return output


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
            dict[str, np.array]: Possible {key :value} outputs
                - {"returns" : .ndarray of returns}
                - {"vol" : np.array of vol}.
        """
        pass

    def init_forecasted(self, T: int) -> dict[str, np.array]:
        """Initialize forecasted attribute.

        Args:
            T (int): Number of forward forecast.

        Returns:
            dict[str, np.array]: {key :value} outputs
                - {"returns" : np.array of returns}
                - {"vol" : np.array of vol}.
        """
        self.forecasted = {"returns": np.zeros((T, self.dim))}
        return self.forecasted

    def record_forecast(self, start_index: int, **forecasts: np.ndarray) -> None:
        """Store the forecast in a pd.DataFrame from a dictonary.

        Args:
            output_dict (dict): The dictonary to store.
            start_index (int): Index corresponding to the series
                starting point in a rolling forecast. E.g. With a 100 rate
                rolling window. `start_index` will increment by 100 on every
                "roll".

        """
        """Record outputs."""
        for forecasts_name in forecasts:
            array_to_add = forecasts[forecasts_name]
            stored_array = self.forecasted[forecasts_name]
            stored_array[
                start_index : start_index + array_to_add.shape[0]
            ] = array_to_add

    def _basic_rolling_split(
        self, dataset: pd.DataFrame, rolling_parameters: list[int]
    ) -> list[int]:
        """Split `dataset` according to `rolling_parameters`."""
        T = np.size(dataset, 0)
        rolling_rate = rolling_parameters[0]
        if rolling_rate <= 0:  # convention: if negative, do not roll
            rolling_rate = T
        lag = rolling_parameters[2]

        q = ceil(T / rolling_rate)

        indices = [0] * (q + 1)
        for i in range(1, q):
            indices[i] = i * rolling_rate - lag
        indices[-1] = np.size(dataset, 0)
        return indices

    def rolling_forecast(
        self,
        series: pd.DataFrame,
        split: list[int],
        rolling_parameters: list[int] = [100, 0, 0],
        id: str = None,
        retrain: bool = True,
    ) -> dict[str, np.array]:
        """Make a rolling forecast on a given dataset/model pair.

        Args:
            series (pd.DataFrame): The series contiaining all the needed data
                split (list[int]): Data split pattern.
            rolling_parameters (optional, list[int]): Parameters on how to
                roll, in order, rolling rate, lag and step size. Rolling
                rate is defined as the length of the interval the model
                needs to forecast before observing the data. Convention is 0
                to never observe data.  Lag and step size are for a more
                advanced usage, thus see the "advanced details"
                documentation.

        Returns:
            dict[str, np.array]: Two standard {key :value} outputs
                - {"returns"  : np.array of returns}
                - {"vol"  : np.array of vol}.

        """
        train_set, test_set = train_test_split(
            series, test_size=split[2], shuffle=False
        )
        indices = self._basic_rolling_split(test_set, rolling_parameters)
        self.init_forecasted(indices[-1])

        # first forecast without using new observations
        self.forecast(train_set, start_index=0, T=indices[1])
        for i in range(1, len(indices) - 1):
            subset = test_set.iloc[indices[i - 1] : indices[i]]
            # use subset to forecast
            self.forecast(subset, start_index=indices[i], T=indices[i + 1] - indices[i],
                          retrain=retrain)

        return self.forecasted

    def get_forecasted(
        self,
        get_the: list = None,
        dtype: type = None,
        id: str = None,
        dates: pd.DatetimeIndex = None,
        dimensions: int = None,
    ) -> dict[str, np.array] | list[float] | np.array | pd.DataFrame:
        """Get subset of forecast from the model in a given format.

        Args:
            get_the (list[str], optional): The name of the outputs you want.
                If "all" is given, returns all of the outputs. It depends on
                the model but two standard `keys` are : {"returns","vol"}.
                Default is to output "all".
            dtype (str, optional): {dict`, list, np.array, pd.DataFrame}
                The way outputs is return. Default is `dict`.
            id (str, optional): Entity id to return.
            dates (pd.DatetimeIndex, optional): Dates to set as index.
            dimensions (int, optional): Dimensions to return.

        Returns:
            dict[str, np.array] | np.array | list[float] | pd.DataFrame:
                The data in the `dtype` format.
        """
        # what to return
        if get_the is None or get_the == "all":
            get_the = self.forecasted.keys()
        elif type(get_the) is str:
            get_the = [get_the]

        # default : return dict
        if dtype is None:
            dtype = dict

        # how to return
        if dtype is dict:
            output = {}
            for get in get_the:
                output[get] = self.forecasted[get]
            return output
        elif dtype is np.array:
            return self.forecasted_to_array(get_the, dtype, id, dates, dimensions)
        elif dtype is list:
            return self.forecasted_to_list(get_the, dtype, id, dates, dimensions)
        # else
        raise ValueError("`dtype` is either dict, list, np.array")

    def forecasted_to_list(
        self,
        get_the: list = None,
        dtype: type = None,
        id: str = None,
        dates: pd.DatetimeIndex = None,
        dimensions: int = None,
    ) -> list[float]:
        """Get subset of outputs from the model in list.

        Args:
            get_the (list[str], optional): The name of the outputs you want.
                If "all" is given, returns all of the outputs. It depends on
                the model but two standard `keys` are : {"returns","vol"}.
                Default is to output "all".
            dtype (str, optional): {dict`, list, np.array, pd.DataFrame}
                The way outputs is return. Default is `dict`.
            id (str, optional): Entity id to return.
            dates (pd.DatetimeIndex, optional): Dates to set as index.
            dimensions (int, optional): Dimensions to return.

        Returns:
            dict[str, np.array] | np.array | list[float] | pd.DataFrame:
                The data in the `dtype` format.
        """
        output = []
        for get in get_the:
            output.append(self.forecasted[get].tolist())
        return output

    def forecasted_to_array(
        self,
        get_the: list = None,
        dtype: type = None,
        id: str = None,
        dates: pd.DatetimeIndex = None,
        dimensions: int = None,
    ) -> np.array:
        """Get subset of outputs from the model in array.

        Args:
            get_the (list[str], optional): The name of the outputs you want.
                If "all" is given, returns all of the outputs. It depends on
                the model but two standard `keys` are : {"returns","vol"}.
                Default is to output "all".
            dtype (str, optional): {dict`, list, np.array, pd.DataFrame}
                The way outputs is return. Default is `dict`.
            id (str, optional): Entity id to return.
            dates (pd.DatetimeIndex, optional): Dates to set as index.
            dimensions (int, optional): Dimensions to return.

        Returns:
            np.array:
                The data in the `dtype` format.
        """
        T = list(self.forecasted.values())[0].shape[0]

        forecasted = np.zeros((T, self.dim * len(get_the)))
        k = 0
        for get in get_the:
            forecasted[:, k : k + self.dim] = self.forecasted[get]
        k += self.dim
        forecasted = np.zeros(T)
        for get in get_the:
            forecasted.append(self.forecasted[get])
        return forecasted


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
