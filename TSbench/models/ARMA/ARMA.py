"""ARMA model."""
import numpy as np
import pandas as pd
from scipy.special import comb

from TSbench.models import Model

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import warnings

warnings.simplefilter("ignore")

class ARMA(Model):
    """Generate outputs using the ARIMA models within a simulation.

    Args:
        p (int): Lag for the autoregressive process.
        d (int): Specifies how many times to difference the time series.
                 Default is 0.
        q (int): Lag for the moving average process.
        ar (np.array): Field to specify weigths of the autoregressive process.
        ma (np.array): Field to specify weigths of the moving average process.
        drift (int): Specifes the drift of the model. Default is 0.
        variance (int): Specifes the variance of the model. Default is 1.
        **model_args: Arguments for `Model`.

    Examples:
        >>> from TSbench.models import ARMA
        >>> arma_model = ARMA(lag=2, dim=2)
        >>> arma_model.generate(T)

    """

    def __init__(
        self,
        p: int = None,
        d: int = 0,
        q: int = None,
        ar: np.array = None,
        ma: np.array = None,
        drift: int = 0,
        variance: int = 1,
        **model_args
    ) -> None:
        """Initialize ARMA."""
        super().__init__(**model_args)

        if self.lag is None and (
            (p is None and ar is None) or (q is None and ma is None)
        ):
            # You need to have `lag` or ((`p` or `ar`) and (`q` or `ma`))").
            # Hence, you take the negation
            raise ValueError(
                "You need to give lag : \n1. Directly with `lag` \n"
                "2. For the AR process with `ar` or `p` "
                "and for the MA process with `ma` or `q`"
            )
        # 1. Directly with `lag`
        elif self.lag is not None:
            p = self.lag
            q = self.lag
        # 2.1 For the AR process with `ar` or `p`
        if ar is None:
            ar = self.rg.uniform(low=-1, high=0, size=(self.dim, p)) / p  # (np.arange(p) + 1)
            ar = np.array(ar, ndmin=2)
            ar = np.hstack((np.ones((self.dim, 1)), ar))
        # 2.2 For the MA process with `ma` or `q`
        if ma is None:
            ma = self.rg.uniform(low=-1, high=0, size=(self.dim, q)) / q  # (np.arange(p) + 1)
            ma = np.array(ma, ndmin=2)
            ma = np.hstack((np.ones((self.dim, 1)), ma))

        self.ar = np.array(ar, ndmin=2)
        self.ar[:, 1:] = -self.ar[:, 1:]
        self.ma = np.array(ma, ndmin=2)
        self.p = np.size(self.ar, 1) - 1
        self.q = np.size(self.ma, 1) - 1

        # upadte lag to reflect model's info
        self.lag = max(self.p, self.q)

        # other parameters
        self.init_length = self.lag
        self.d = d
        self.sigma = self.corr_mat.mat
        self.drift = drift
        self.variance = variance

    def generate(self, T: int) -> dict[str, np.array]:
        """Set self.outputs to `T` generated values using ARMA.

        Args:
            T (int): Number of observations to generate.

        Returns:
            dict[str, np.array] : {"returns" : np.array of returns}.
        """
        # initialization
        x = np.zeros((self.dim, T))
        z = self.rg.standard_normal(size=(self.dim, T))

        #x[:, 0] = z[:, 0]
        # initialize MA
        for t in range(0, self.q + 1):
            for k in range(0, self.dim):
                x[k, t] = self.flipped_dot(self.ma[k, 0 : t + 1], z[k, 0 : t + 1])

        # generate MA
        for t in range(self.q + 1, T):
            for k in range(0, self.dim):
                x[k, t] = self.flipped_dot(self.ma[k, :], z[k, t - self.q : t + 1])
        #x = z

        # initialize AR
        for t in range(1, self.p + 1):
            for k in range(0, self.dim):
                x[k, t] = self.flipped_dot(self.ar[k, 0 : t + 1], x[k, 0 : t + 1])

        # generate AR
        for t in range(self.p + 1, T):
            x[:, t] = (
                self.flipped_dot_dimension_wise(self.ar[:, :], x[:, t - self.p : t + 1])
            )  # x_{t-i-j}

        self.record_outputs(returns=np.transpose(x))
        return self.outputs

    def flipped_dot_dimension_wise(self, a: np.array, b: np.array) -> np.array:
        """Calculate the dot product of two vectors.

        For every dimension, the second vector is flipped which means
        that the entry are read from right to left.

        Args:
            a (np.array): First array.
            b (np.array): Second array, with entry to flip for each dimension.

        Returns:
            np.array : Flipped dot product dimension-wise.

        """
        y = np.zeros(self.dim)
        for dim in range(0, self.dim):
            y[dim] = np.dot(a[dim, :], b[dim, ::-1])
        return y

    def flipped_dot(self, a: np.array, b: np.array) -> np.array:
        """Calculate the dot product of two vectors.

        The second vector is flipped which means that the entry are read
        from right to left.

        Args:
            a (np.array): First array.
            b (np.array): Second array, with entry to flip.

        Returns:
            np.array : Flipped dot product.

        """
        return np.dot(a, b[::-1])

    def generate_ar(self, x: np.array, t: int) -> np.array:
        """Generate the autoregressive process with lagged values up until `self.p`.

        Args:
            x (np.array): Array of lagged values dimension-wise.
            t (np.array): The current time.

        Returns:
            np.array : Generated autoregressive process.

        """
        for k in range(0, self.dim):
            x[k, t] = np.dot(self.ar[k, :], x[k, (t - self.p) : t + 1])
        return x[:, t]

    def generate_ma(self, z: np.array, t: int = None) -> np.array:
        """Generate the moving average process with lagged values up until `self.q`.

        Args:
            x (np.array): Array of lagged noise dimension-wise.
            t (np.array): The current time.

        Returns:
            np.array : Generated moving average process.

        """
        y = np.zeros(self.dim)
        for k in range(0, self.dim):
            y[k] = np.dot(self.ma[k, :], z[k, ::-1])
        return y

    def diff_x(self, x: np.array, t: int = None) -> np.array:
        """Generate the `self.d` level differenciation of the input `x`.

        Compute for current time.

        Args:
            x (np.array): Array of current values.
            t (np.array): The current time.

        Returns:
            np.array : Generated differenciated input values.

        """
        if np.size(x) > 0:
            y = np.zeros(self.dim)
            alpha = [(-1) ** j * comb(self.d, j) for j in range(1, self.d + 1)]
            for k in range(0, self.dim):
                y[k] = np.dot(alpha, x[k, ::-1])
            return y
        else:
            return 0

    def diff_ar(self, x: np.array, t: int = None) -> np.array:
        """Generate the `self.d` level differenciation.

        Use lagged values up until `self.p`.

        Args:
            x (np.array): Array of lagged values dimension-wise.
            t (np.array): The current time.

        Returns:
            np.array : Generated differenciated lagged values.

        """
        if np.size(x) > 0:
            diff = np.zeros(self.dim)
            for k in range(0, self.dim):
                for i in range(1, self.p + 1):
                    for j in range(1, self.d + 1):
                        diff[k] += (
                            (-1) ** j
                            * comb(self.d, j)
                            * self.ar[k, i]
                            * x[k, np.size(x, 0) - i - j]
                        )
            return diff
        else:
            return 0

    def __repr__(self) -> str:
        """ARMA info and dimension."""
        name = super().__str__()
        info = (
            "("
            + str(self.p)
            + ", "
            + str(self.d)
            + ", "
            + str(self.q)
            + ", dim="
            + str(self.dim)
            + ")"
        )
        return name + info

    def params(self) -> dict[str, any]:
        """Parameters dictonary of the ARMA model.

        Returns:
            dict[str, any] : Parameters of the ARMA model.
        """
        return {
            "dim": self.dim,
            "lag": self.lag,
            "mat": self.corr_mat.mat,
            "drift": self.drift,
            "variance": self.variance,
            "ar": self.ar,
            "d": self.d,
            "ma": self.ma,
        }

    def train(self, series: pd.DataFrame) -> "ARMA":
        """Train model using `series` as the trainning set.

        Args:
            series (pd.DataFrame): Input series.

        Returns:
            Model: Trained ARMA model.

        """
        if self.dim == 1:
            self.sm_arma = ARIMA(series, order=(self.p, self.d, self.q)).fit()
            self.sm_arma.remove_data()
        else:
            self.sm_arma = sm.tsa.VARMAX(series, order=(self.p, self.q)).fit(disp=False)
            # `remove_data` cause en error whenever apply is reused later
            # to get the data back. This is why this is used instead.
            removed = np.empty((1, self.dim), dtype=int)
            self.sm_arma = self.sm_arma.apply(removed)
        return self

    def forecast(
        self, series: pd.DataFrame = None, start_index: int = None, T: int = None, retrain: bool = False
    ) -> dict[str, np.array]:
        """Forecast a timeseries.

        Knowing `series` from `start_index`, set self.forecasted to `T`
        forecasted values.

        Args:
            series (pd.DataFrame): Input series.
            start_index (int): Index corresponding to the series.
                starting point in a rolling forecast. E.g. With a 100 rate
                rolling window. `start_index` will increment by a 100 on
                every "roll".
            T (int): Number of forward forecast.

        Returns:
            dict[str, np.array]: Possible {key :value} outputs
                - {"returns" : np.array of returns}
                - {"vol" : np.array of vol}.
        """
        # update model with most recent available series
        self.append(series)

        forecast = self.sm_arma.forecast(T)

        # numpy idiosyncratic detail :
        # (n,) array \neq (n,1) array, hence the reshape
        forecast = forecast.reshape(-1, self.dim)

        self.record_forecast(start_index, returns=forecast)
        return self.forecasted

    def append(self, series):
        """Append a serie to statsmodel endog variable.

        Args:
            series (pd.DataFrame): Input series.
        """
        if (self.sm_arma.model.endog is None or  # more complex multivariate
                self.sm_arma.model.endog.dtype == "int64"):
            self.sm_arma = self.sm_arma.apply(endog=series.to_numpy())
        else:
            self.sm_arma = self.sm_arma.append(endog=series.to_numpy())
