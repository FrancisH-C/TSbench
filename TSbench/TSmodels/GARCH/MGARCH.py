from __future__ import annotations
import numpy as np
from numpy import random as rand
from numpy.linalg import cholesky

from TSbench.TSmodels.models import GeneratorModel
from TSbench.TSmodels.GARCH import GARCH

from typing import Callable


class VEC_GARCH(GARCH):
    """Generate outputs using the VEC-GARCH models within a simulation.

    Args:
        **garch_args: Arguments for `GARCH`. See `GARCH` documentation for a list
            of all supported keywords.
    """

    def __init__(self, **garch_args) -> None:
        """Initialize VEC_GARCH."""
        super().__init__(**garch_args)

    def generate(self, T: int) -> dict[str, np.array]:
        """Generate `T` values using VEC-GARCH.

        Args:
            T (int): Number of observations to generate.

        Returns:
            dict[str, np.array] : {key :value} outputs
                - {"returns"  : np.array of returns}
                - {"vol"  : np.array of vol}.
        """
        # initialization
        epsilon, vol = self.initial_state_default(T)
        z = rand.standard_normal(size=(self.dim, T))

        # generate observations
        for t in range(self.init_length(), T):
            vol[t, :, :] = (
                self.C + self.generate_ma(epsilon, t) + self.generate_ar(vol, t)
            )
            vol[t, :, :] = cholesky(vol[t, :, :])
            epsilon[:, t] = np.dot(vol[t, :, :], z[:, t])

        self.outputs = {"returns": epsilon, "vol": vol}
        return self.outputs

    def initial_state_default(self, T: int) -> tuple[np.array, np.array]:
        """Generate inital state as zero returns and identity variance matrix.

        Args:
            T (int): Number of observations to generate.

        Returns:
            tuple[np.array,np.array] : A tupple (returns, vol).

        """
        epsilon = np.zeros((self.dim, T))
        vol = np.zeros((T, self.dim, self.dim))
        for t in range(T):
            vol[t, :, :] = np.eye(self.dim, self.dim)
        return epsilon, vol

    def generate_ar(self, vol: np.array, t: int) -> np.array:
        """Generate the VEC-GARCH autoregressive process.

        Use up until `self.p`.

        Args:
            vol (np.array): Array of lagged vol dimension-wise.
            t (np.array): The current time.

        Returns:
            np.array : Generated autoregressive process.

        """
        y = np.zeros((self.dim, self.dim))
        for j in range(0, self.p):
            y += np.matmul(self.B[j, :, :], vol[t - j - 1, :, :])
        return y

    def generate_ma(self, epsilon: np.array, t: int) -> np.array:
        """Generate the VEC-GARCH moving average process.

        Use up until `self.q`.

        Args:
            epsilon (np.array): Array of lagged returns dimension-wise.
            t (np.array): The current time.

        Returns:
            np.array : Generated moving average process.

        """
        y = np.zeros((self.dim, self.dim))
        for j in range(0, self.q):
            y += np.matmul(
                self.A[j, :, :], np.outer(epsilon[:, t - j - 1], epsilon[:, t - j - 1])
            )
        return y


class SPD_VEC_GARCH(VEC_GARCH):
    """Generate outputs using the SPD-VEC-GARCH models within a simulation.

    Args:
        **garch_args: Arguments for `GARCH`. See `GARCH` documentation for a list
            of all supported keywords.

    """

    def __init__(self, **garch_args) -> None:
        """Initialize SPC_VEC_GARCH."""
        super().__init__(**garch_args)

    def generate(self, T: int) -> dict[str, np.array]:
        """Generate `T` values using SPD-VEC-GARCH.

        Args:
            T (int): Number of observations to generate.

        Returns:
            dict[str, np.array] : {key :value} outputs :
                - {"returns"  : np.array of returns}
                - {"vol"  : np.array of vol}
        """
        # initialization
        epsilon, vol = self.initial_state_default(T)
        z = rand.standard_normal(size=(self.dim, T))

        # generate observations
        translations = 0
        for t in range(self.init_length(), T):
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
                    np.eye(np.size(vol, 1)) + 2 * 10 ** (-6)
                )
                translations += 1

            vol[t, :, :] = cholesky(vol[t, :, :])
            epsilon[:, t] = np.matmul(vol[t, :, :], z[:, t])
        if translations > 0:
            print(
                "Number of translations to make matrix positive definite :",
                translations,
            )

        self.outputs = {"returns": epsilon, "vol": vol}
        return self.outputs

    def generate_ar(self, vol: np.array, t: int) -> np.array:
        """Generate the VEC-GARCH autoregressive process.

        Use up until `self.p`.

        Args:
            vol (np.array): Array of lagged vol dimension-wise.
            t (np.array): The current time.

        Returns:
            np.array : Generated autoregressive process.

        """
        y = np.zeros((self.dim, self.dim))
        for j in range(0, self.p):
            y += np.matmul(self.B[j, :, :], vol[t - j - 1, :, :])
        return y

    def generate_ma(self, epsilon: np.array, t: int) -> np.array:
        """Generate the SPD-VEC-GARCH moving average process.

        Use up until `self.q`.

        Args:
            epsilon (np.array): Array of lagged returns dimension-wise.
            t (np.array): The current time.

        Returns:
            np.array : Generated moving average process.

        """
        y = np.zeros((self.dim, self.dim))
        for j in range(0, self.q):
            y += np.dot(
                self.A[j, :, :], np.outer(epsilon[:, t - j - 1], epsilon[:, t - j - 1])
            )
        return y
