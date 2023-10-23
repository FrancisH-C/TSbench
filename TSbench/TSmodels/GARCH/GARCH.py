"""GARCH model."""
from __future__ import annotations
import numpy as np
from numpy import random as rand

from TSbench.TSmodels.models import GeneratorModel


class GARCH(GeneratorModel):
    """Generate outputs using the GARCH models within a simulation.

    Args:
        A (np.array, optional): Field to specify weigths of the
            moving average process.
        B (np.array, optional): Field to specify weigths of the
                autoregressive process.
        C (np.array, optional): Field to specify constant to add to the
            generated variance. Needs to be positive definite. Default is the
            identity matrix.
        drift (int, optional): Specifes the drift of the model. Default is 0.
        **model_args: Arguments for `models` `Model`.
            Possible keywords are `dim`, `lag` and `corr_mat`.

    """

    def __init__(
        self,
        A: np.array = None,
        B: np.array = None,
        C: np.array = None,
        drift: int = 0,
        **model_args,
    ) -> None:
        """Initialize GARCH."""
        super().__init__(**model_args, default_features=["returns", "vol"])
        if len(self.feature_label) != 2 * self.dim:
            raise ValueError("Need 'feature_label' with `self.dim + 1` entries")

        # parameters for the model
        if self.lag is None and (A is None or B is None):
            # You need to have `lag` or `A` and `B`
            # Hence, you take the negation
            raise ValueError(
                "You need to give lag: Directly with `lag` or with `A` and `B`"
            )
        elif self.lag is None:
            p = np.size(A, 0)
            q = np.size(B, 0)
            self.lag = max(p, q)
            # a tester pour dim > 1

        # A or B can be defined, you check it individually
        if A is None:
            if self.dim == 1:
                A = rand.uniform(size=(self.lag)) / self.lag
                # A = rand.uniform(low=-1, size=(lag)) / lag
            else:
                A = rand.uniform(size=(self.lag, self.dim, self.dim)) / (
                    np.sqrt(9 * self.lag)
                )
            # A = rand.uniform(low=-1, size=(dim, lag, lag)) / lag
        if B is None:
            if self.dim == 1:
                B = rand.uniform(size=(self.lag)) / self.lag
            else:
                B = rand.uniform(size=(self.lag, self.dim, self.dim)) / (
                    np.sqrt(9 * self.lag)
                )

        if self.dim == 1:
            self.A = np.array(A)
            self.B = np.array(B)
            self.C = np.array(C)
        else:
            self.A = np.array(A, ndmin=3)
            self.B = np.array(B, ndmin=3)
            self.C = np.array(C, ndmin=3)

        self.drift = drift

        self.q = np.size(self.A, 0)
        self.p = np.size(self.B, 0)

        if C is None:
            # self.C = np.eye(self.dim)
            self.C = self.corr_mat.mat
        else:
            self.C = C

    def generate(self,  T: int, timeseries: tuple(np.array, np.array) = None, collision: str ="overwrite") -> tuple(np.array, np.array):
        """Generate `T` values using GARCH.

        Args:
            T (int): Number of observations to generate.

        Returns:
            dict[str, np.array] : {key :value} outputs
                - {"returns"  : np.array of returns}
                - {"vol"  : np.array of vol}.
        """
        self.set_timeseries(timeseries=timeseries)
        epsilon = np.zeros(T)
        vol = np.ones(T)
        z = rand.standard_normal(T)

        # generate
        for t in range(self.init_length(), T):
            vol[t] = np.sqrt(
                self.C + self.generate_ma(epsilon, t) + self.generate_ar(vol, t)
            )
            epsilon[t] = vol[t] * z[t]

        return self.set_timeseries(observations=[epsilon.reshape(T, 1), vol.reshape(T, 1)], collision=collision)

    def generate_ar(self, vol: np.array, t: int) -> np.array:
        """Generate the GARCH autoregressive process with lagged values.

        Use up until `self.p`.

        Args:
            vol (np.array): Array of lagged vol dimension-wise.
            t (np.array): The current time.

        Returns:
            np.array : Generated autoregressive process.

        """
        y = 0
        for j in range(0, self.p):
            y += np.dot(self.B[j], vol[t - j - 1])
        return y

    def generate_ma(self, epsilon: np.array, t: int) -> np.array:
        """Generate the GARCH moving average process with lagged values.

        Use up until `self.q`.

        Args:
            epsilon (np.array): Array of lagged returns dimension-wise.
            t (np.array): The current time.

        Returns:
            np.array : Generated moving average process.

        """
        y = 0
        for j in range(0, self.q):
            y += np.dot(
                self.A[j], np.dot(epsilon[t - j - 1], np.transpose(epsilon[t - j - 1]))
            )
        return y

    def __repr__(self) -> str:
        """ARMA info and dimension."""
        name = super().__str__()
        info = "(" + str(self.q) + ", " + str(self.p) + ")"
        return name + info

    #    def __repr__(self) -> str:
    #        """GARCH info and dimension"""
    #        name = super().__str__()
    #        info = "(" + str(self.p) + ", " + str(self.q) + ")" + ", "
    #        dim = "dim = " + str(self.dim)
    #        parameters = " with parameters: " + "\nA = "  + \
    #                    str(self.A) + "\n\nB = " + \
    #            str(self.B) + "\n\nC = " + str(self.C)
    #        return name + info + dim + parameters
