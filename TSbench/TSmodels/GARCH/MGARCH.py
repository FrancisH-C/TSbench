from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy import random as rand
from numpy.linalg import cholesky

from TSbench.TSdata.data import AnyData
from TSbench.TSmodels.GARCH import GARCH
from TSbench.TSmodels.models import GeneratorModel


class VEC_GARCH(GARCH):
    """Generate outputs using the VEC-GARCH models within a simulation.

    Args:
        **garch_args: Arguments for `GARCH`. See `GARCH` documentation for a list
            of all supported keywords.
    """

    def __init__(self, **garch_args) -> None:
        """Initialize VEC_GARCH."""
        super().__init__(**garch_args)

    def generate(
        self, N: int, reset_timestamp: bool = True, collision: str = "overwrite"
    ) -> AnyData:
        """Generate `N` values using VEC-GARCH.

        Args:
            N (int): Number of observations to generate.

        Returns:
            dict[str, np.ndarray] : {key :value} outputs
                - {"returns"  : np.ndarray of returns}
                - {"vol"  : np.ndarray of vol}.
        """
        # initialization
        epsilon = np.zeros((N, self.dim))
        vol = np.zeros((N, self.dim, self.dim))
        for t in range(N):
            vol[t, :, :] = np.eye(self.dim, self.dim)
        z = rand.standard_normal(size=(N, self.dim))

        # generate observations
        for t in range(self.lag, N):
            vol[t, :, :] = (
                self.C + self.generate_ma(epsilon, t) + self.generate_ar(vol, t)
            )
            vol[t, :, :] = cholesky(vol[t, :, :])
            epsilon[t, :] = np.dot(vol[t, :, :], z[t, :])

        return self.set_data(
            data=[epsilon, vol], reset_timestamp=reset_timestamp, collision=collision
        )

    def generate_ar(self, vol: np.ndarray, t: int) -> np.ndarray:
        """Generate the VEC-GARCH autoregressive process.

        Use up until `self.p`.

        Args:
            vol (np.ndarray): Array of lagged vol dimension-wise.
            t (np.ndarray): The current time.

        Returns:
            np.ndarray : Generated autoregressive process.

        """
        y = np.zeros((self.dim, self.dim))
        for j in range(0, self.p):
            y += np.matmul(self.B[j, :, :], vol[t - j - 1, :, :])
        return y

    def generate_ma(self, epsilon: np.ndarray, t: int) -> np.ndarray:
        """Generate the VEC-GARCH moving average process.

        Use up until `self.q`.

        Args:
            epsilon (np.ndarray): Array of lagged returns dimension-wise.
            t (np.ndarray): The current time.

        Returns:
            np.ndarray : Generated moving average process.

        """
        y = np.zeros((self.dim, self.dim))
        for j in range(0, self.q):
            y += np.matmul(
                self.A[j, :, :], np.outer(epsilon[t - j - 1, :], epsilon[t - j - 1, :])
            )
        return y


class VEC_SPD_GARCH(VEC_GARCH):
    """Generate outputs using the SPD-VEC-GARCH models within a simulation.

    Args:
        **garch_args: Arguments for `GARCH`. See `GARCH` documentation for a list
            of all supported keywords.

    """

    def __init__(self, **garch_args) -> None:
        """Initialize VEC_SPC_GARCH."""
        super().__init__(**garch_args)

    def generate(
        self,
        N: int,
        reset_timestamp: bool = True,
        collision: str = "overwrite",
        verbose: bool = False,
    ) -> AnyData:
        """Generate `N` values using VEC-SPD-GARCH.

        Args:
            N (int): Number of observations to generate.

        Returns:
            dict[str, np.ndarray] : {key :value} outputs :
                - {"returns"  : np.ndarray of returns}
                - {"vol"  : np.ndarray of vol}
        """
        # initialization
        epsilon = np.zeros((N, self.dim))
        vol = np.zeros((N, self.dim, self.dim))
        for t in range(N):
            vol[t, :, :] = np.eye(self.dim, self.dim)
        z = rand.standard_normal(size=(N, self.dim))

        # generate observations
        translations = 0
        for t in range(self.lag, N):
            vol[t, :, :] = (
                self.C + self.generate_ma(epsilon, t) + self.generate_ar(vol, t)
            )

            # Symmetrize
            vol[t, :, :] = (vol[t, :, :] + np.transpose(vol[t, :, :])) / 2

            # Transalte to make positive definite
            eigenvalues, v = np.linalg.eigh(vol[t, :, :])
            if eigenvalues[0] < 0:
                # add the smallest egeinvalue to diagonal plus an epsilon to account for
                # machine error.
                vol[t, :, :] -= eigenvalues[0] * (
                    np.eye(np.size(vol, 1)) + 2  # * 10 ** (-6)
                )
                translations += 1

            vol[t, :, :] = cholesky(vol[t, :, :])
            epsilon[t, :] = np.matmul(vol[t, :, :], z[t, :])
        if translations > 0 and verbose:
            print(
                "Number of translations to make matrix positive definite :",
                translations,
            )

        return self.set_data(
            data=[epsilon, vol], reset_timestamp=reset_timestamp, collision=collision
        )

    def generate_ar(self, vol: np.ndarray, t: int) -> np.ndarray:
        """Generate the VEC-GARCH autoregressive process.

        Use up until `self.p`.

        Args:
            vol (np.ndarray): Array of lagged vol dimension-wise.
            t (np.ndarray): The current time.

        Returns:
            np.ndarray : Generated autoregressive process.

        """
        y = np.zeros((self.dim, self.dim))
        for j in range(0, self.p):
            y += np.matmul(self.B[j, :, :], vol[t - j - 1, :, :])
        return y

    def generate_ma(self, epsilon: np.ndarray, t: int) -> np.ndarray:
        """Generate the VEC-SPD-GARCH moving average process.

        Use up until `self.q`.

        Args:
            epsilon (np.ndarray): Array of lagged returns dimension-wise.
            t (np.ndarray): The current time.

        Returns:
            np.ndarray : Generated moving average process.

        """
        y = np.zeros((self.dim, self.dim))
        for j in range(0, self.q):
            y += np.dot(
                self.A[j, :, :], np.outer(epsilon[t - j - 1, :], epsilon[t - j - 1, :])
            )
        return y

    def __repr__(self) -> str:
        """VGARCH info and dimension."""
        name = super().__str__()
        info = "(" + str(self.p) + ", " + str(self.q) + ", dim=" + str(self.dim) + ")"
        return name + info


class DCC_GARCH(GeneratorModel):
    """Generate outputs using the DCC-GARCH models within a simulation.

    Args:
        update_rule (Callable[...,int], optional): How to update the correlation matrix
             at each timestep. Default is `self.DCCE`.
        univariate (list[GARCH], optional): List of `self.dim` GARCH-type model used to
             generate the univariate variance. Default is to have a list of
            `self.dim` `GARCH` model.
        R (np.ndarray, optional): A correlation matrix. Default is the identity matrix.
        **model_args: Arguments for `Model`. Possible keywords are `dim`, `lag`
            and `corr`.
    """

    def __init__(
        self,
        update_rule: Optional[Callable[..., int]] = None,
        univariate: Optional[list[GARCH]] = None,
        R: Optional[np.ndarray] = None,
        theta1: float = 0.05,
        theta2: float = 0.05,
        **model_args,
    ) -> None:
        """Initialize DCC_GARCH."""
        super().__init__(**model_args)

        if univariate is None:
            self.univariate = [GARCH(lag=self.lag) for _ in range(0, self.dim)]
        else:
            self.univariate = univariate

        if R is None:
            self.R = self.corr_mat.mat
        else:
            self.R = R

        if update_rule is None:
            # lambda function can't be pickle so theta can't be pass as parameter
            self.update_rule = self.DCCE
            self.theta1 = theta1
            self.theta2 = theta2
        else:
            self.update_rule = update_rule

    def generate(
        self, N: int, reset_timestamp: bool = True, collision: str = "overwrite"
    ) -> AnyData:
        """Generate `N` values using DCC-GARCH.

        Args:
            N (int): Number of observations to generate.

        Returns:
            dict[str, np.ndarray]: {key :value} outputs
                - {"returns"  : np.ndarray of returns}
                - {"vol"  : np.ndarray of vol}.
        """
        # initialization
        epsilon = np.zeros((self.dim, N))
        vol = np.zeros((N, self.dim, self.dim))
        z = rand.standard_normal(size=(self.dim, N))

        # generate univariate
        for i in range(0, len(self.univariate)):
            vol[:, i, i] = self.univariate[i].generate(N)
            # x, vol[:, i, i] = outputs["returns"], outputs["vol"]
            # x = outputs["returns"]
            # vol[:, i, i] = outputs["vol"].reshape(N)  # from (T, 1)

        # compose
        R0 = self.R
        for t in range(2, N):
            vol[t, :, :] = np.matmul(vol[t, :, :], np.matmul(self.R, vol[t, :, :]))
            self.update_rule(z=z, R0=R0, t=t, theta1=self.theta1, theta2=self.theta2)

            vol[t, :, :] = cholesky(vol[t, :, :])

            epsilon[:, t] = np.matmul(vol[t, :, :], z[:, t])

        return self.set_data(
            data=[epsilon, vol], reset_timestamp=reset_timestamp, collision=collision
        )

    def __repr__(self) -> str:
        """GARCH info by dimension."""
        name = super().__repr__()
        output = " with"
        for garch in self.univariate:
            output = output + ", " + str(garch)

        return name + output

    def DCCE(
        self,
        z: np.ndarray,
        R0: np.ndarray,
        t: int,
        theta1: float = 0.05,
        theta2: float = 0.05,
    ) -> None:
        """Update correlation matrix using DCC_E (Engle 2000).

        Args:
            z (np.ndarray): Observed noise, usually a Normal(0,1).
            R0 (np.ndarray): Initial correlation matrix.
            t (np.ndarray): The current time.

        """
        m = 2
        u = z[:, t - m : t]
        psi = np.eye(self.dim)
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                psi[i, j] = np.matmul(u[i, :], u[j, :]) / np.sqrt(
                    np.linalg.norm(u[i, :]) ** 2 * np.linalg.norm(u[j, :]) ** 2
                )
        self.R = np.dot(1 - theta1 - theta2, R0) + theta1 * psi + np.dot(theta2, self.R)
