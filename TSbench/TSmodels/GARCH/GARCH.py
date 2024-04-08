#!/usr/bin/env python
from __future__ import annotations

from typing import Optional

import numpy as np

from TSbench.TSdata.data import AnyData
from TSbench.TSmodels.models import GeneratorModel


class GARCH(GeneratorModel):
    """Generate outputs using the GARCH models within a simulation.

    Args:
        A (np.ndarray, optional): Field to specify weigths of the
            moving average process.
        B (np.ndarray, optional): Field to specify weigths of the
                autoregressive process.
        C (np.ndarray, optional): Field to specify constant to add to the
            generated variance. Needs to be positive definite. Default is the
            identity matrix.
        drift (int, optional): Specifes the drift of the model. Default is 0.
        **model_args: Arguments for `models` `Model`.
            Possible keywords are `dim`, `lag` and `corr_mat`.

    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    drift: int

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        C: Optional[np.ndarray] = None,
        drift: int = 0,
        **model_args,
    ) -> None:
        """Initialize GARCH."""
        super().__init__(**model_args)

        # parameters for the model
        if self.lag is None and (A is None or B is None):
            # You need to have `lag` or (`A` and `B`)
            raise ValueError(
                "You need to give lag: Directly with `lag` or with `A` and `B`"
            )
        # if self.lag is None:
        #     p = np.size(A, 0)
        #     q = np.size(B, 0)
        #     self.lag = max(p, q)
        #     # a tester pour dim > 1

        # Initialize A with lag
        if A is None:
            if self.dim == 1:
                A = self.rg.uniform(size=(self.lag)) / self.lag
            else:
                A = self.rg.uniform(size=(self.lag, self.dim, self.dim)) / (
                    np.sqrt(9 * self.lag)
                )

        # Initialize B with lag
        if B is None:
            if self.dim == 1:
                B = self.rg.uniform(size=(self.lag)) / self.lag
            else:
                B = self.rg.uniform(size=(self.lag, self.dim, self.dim)) / (
                    np.sqrt(9 * self.lag)
                )

        if self.dim == 1:
            self.A = np.array(A)
            self.B = np.array(B)
        else:
            self.A = np.array(A, ndmin=3)
            self.B = np.array(B, ndmin=3)

        # upate p, q and lag
        self.q = np.size(self.A, 0)
        self.p = np.size(self.B, 0)
        self.lag = max(self.p, self.q)

        # Initialize C and drift
        if C is None:
            self.C = self.corr_mat.mat
        else:
            self.C = C
        self.drift = drift

    def set_feature_label(self, feature_label: Optional[list[str]] = None) -> None:
        if feature_label is None:
            if self.dim > 1:
                feature_label = ["retuns"] + ["vol" + str(i) for i in range(self.dim)]
            else:
                feature_label = ["retuns"] + ["vol"]
        if len(feature_label) != self.dim + 1:
            raise ValueError("Need 'feature_label' with `self.dim + 1` entries")
        self.feature_label = np.array(feature_label)

    def generate(
        self, N: int, reset_timestamp: bool = True, collision: str = "overwrite"
    ) -> AnyData:
        """Generate `N` values using GARCH.

        Args:
            N (int): Number of observations to generate.

        Returns:
            dict[str, np.ndarray] : {key :value} outputs
                - {"returns"  : np.ndarray of returns}
                - {"vol"  : np.ndarray of vol}.
        """
        # initial value
        initial = self.get_data(tstype=np.ndarray)
        epsilon = np.zeros((N - initial.shape[0], self.dim))
        vol = np.zeros((N - initial.shape[0], self.dim))

        if initial.size != 0:
            # add initial to epsilon and vol
            epsilon = np.concatenate((initial[:, :, 0], epsilon), axis=0)
            vol = np.concatenate((initial[:, :, 1], vol), axis=0)

        z = self.rg.standard_normal(N)

        # generate
        for t in range(0, N):
            vol[t] = np.sqrt(
                (self.C + self.generate_ma(epsilon, t) + self.generate_ar(vol, t))[0, 0]
            )
            epsilon[t] = vol[t] * z[t]

        return self.set_data(
            data=[epsilon.reshape(N, 1), vol.reshape(N, 1)],
            reset_timestamp=reset_timestamp,
            collision=collision,
        )

    def generate_ar(self, vol: np.ndarray, t: int) -> np.ndarray:
        """Generate the GARCH autoregressive process with lagged values.

        Use up until `self.p`.

        Args:
            vol (np.ndarray): Array of lagged vol dimension-wise.
            t (np.ndarray): The current time.

        Returns:
            np.ndarray : Generated autoregressive process.

        """
        y = np.zeros(1)
        for j in range(0, self.p):
            y += np.dot(self.B[j], vol[t - j - 1])
        return y

    def generate_ma(self, epsilon: np.ndarray, t: int) -> np.ndarray:
        """Generate the GARCH moving average process with lagged values.

        Use up until `self.q`.

        Args:
            epsilon (np.ndarray): Array of lagged returns dimension-wise.
            t (np.ndarray): The current time.

        Returns:
            np.ndarray : Generated moving average process.

        """
        y = np.zeros(1)
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
