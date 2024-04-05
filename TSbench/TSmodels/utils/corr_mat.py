"""Correlation matrix wrapper for numpy array."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.random import PCG64, Generator


class Corr_mat:
    """This class is still really basic and should be improved.

    It wraps a `np.ndarrray` as its attribute `mat` which represents
    as a correlation matrix.  The aim of this is to make it easier
    to define, generate and modify correlation matrices. It achieves
    it with predefined correlation matrix templates and
    generators. Moreover, any modifications to the `np.ndarray`
    is making sure that it keeps being well defined as a correlation
    matrix.

    """

    mat: np.ndarray
    rg: Generator
    dim: int
    method: Callable[..., np.ndarray]
    method_args: dict

    def __init__(
        self,
        mat: Optional[np.ndarray] = None,
        rg: Optional[Generator] = None,
        dim: Optional[int] = None,
        method: Optional[Callable[..., np.ndarray]] = None,
        **method_args,
    ) -> None:
        """Initialize the correlation matrix.

        With either a given matrix `mat` or with with a `method` to generate a `dim`
        by `dim` matrix.

        Implemented methods are :

            {"uncorrelated", "random", "const", "specific_corr"}

        Args:
            mat (np.ndarray, optional): An already defined correlation matrix.
                Default is None.
            dim (int, optional): The corralation matrix is `dim` by `dim`. If `mat` is
                defined, it uses it to defined its dimension.
            method ({"uncorrelated", "random", "const", "specific_corr"}, optional):
                The method to use to generate a random correlation matrix.
            **method_const: Any constant parameters needed for `method`.

        """
        # initialize
        if mat is not None and dim is None:
            dim = np.size(mat, 0)
        elif mat is None and dim is not None:
            self.dim = dim
        else:
            raise ValueError("Give exactly one of {`dim`, `mat`} parameters.")

        if rg is None:
            rg = Generator(PCG64())
        self.rg = rg

        self.make_corr_mat = self.normalize_correlation_matrix
        self.set_method(method)

        self.set_mat(**method_args)

    def set_dim(self, dim: int) -> None:
        """Set dimension for correlation matrix.

        Args:
            dim (int): new dimension to set.
        """
        self.dim = dim

    def set_method(
        self, method: Optional[Callable[..., np.ndarray]] = None, **method_args
    ) -> None:
        """A method takes **method_args and returns a well-define correlation matrix.

        Args:
            method (int): new method to use.
        """
        if method is None:
            method = self.distribution_method
        self.method = method
        self.method_args = method_args

    def set_mat(self) -> None:
        """Set new correlation matrix using `self.mat`.

        Args:
            **kwargs: keyword arguments used in `self.method`.
        """
        self.mat = self.method(**self.method_args)

    def is_corr_mat(self) -> bool:
        """Test on matrix a correlation matrix.

        Returns:
            bool: returns if the matrix a correlation matrix.
        """
        # One on the diagonal
        for k in range(self.dim):
            if self.mat[k, k] != 1:
                return False

        for i in range(self.dim):
            for j in range(self.dim):
                if self.mat[i, j] > 1:
                    return False
        # It passes all tests
        return True

    def normalize_correlation_matrix(self) -> np.ndarray:
        """Make sure the matrix is a correlation matrix.

        The operations are :

        1. Normalize column vector
        2. Put one diagonal
        3. Symmetrize the matrix
        """
        # Normalize column vector if an entry is more than 1
        for j in range(0, self.dim):
            vec = self.mat[:, j]
            if max(abs(vec)) > 1:
                self.mat[j, j] = 0
                self.mat[:, j] /= np.linalg.norm(vec)

        # One on the diagonal
        for k in range(self.dim):
            self.mat[k, k] = 1

        # symmetrize
        self.mat = (self.mat + self.mat.T) / 2
        return self.mat

    def uncorrelated(self) -> np.ndarray:
        """Reset the correlation matrix to identity."""
        self.mat = np.eye(self.dim)
        return self.mat

    def distribution_method(
        self, distribution_name: Optional[str] = None, **distribution_args
    ):
        if distribution_name is None:
            distribution_name = "uniform"
        if not distribution_args:
            distribution_args = {"low": -1, "high": 1}
        distribution = getattr(self.rg, distribution_name)

        def mat_from_distribution(
            distribution: Callable[..., float], **distribution_args
        ) -> np.ndarray:
            """Generate a correlation matrix from a distribution function."""
            mat = np.eye(self.dim)
            for i in range(np.size(mat, 0)):
                for j in range(np.size(mat, 1)):
                    mat[i, j] = distribution(**distribution_args)

            for i in range(np.size(mat, 0)):
                mat[i, i] = 1
            self.mat = mat
            self.make_corr_mat()
            return self.mat

        return mat_from_distribution(distribution, **distribution_args)

    def const_method(self, corr: float = 0.5) -> np.ndarray:
        """Generate a correlation matrix with all entries being a constant.

        Args:
            corr (int, optional): constant correlation

        Returns:
            np.ndarray: A constant correlation matrix

        """
        # Resets the correlation matrix
        self.mat = corr * np.ones((self.dim, self.dim))
        self.make_corr_mat()
        return self.mat

    def specific_corr_method(self, indices: list[tuple], corr: list[int]) -> np.ndarray:
        """Set the matrix at `corr` for `indices`.

        Args:
            indices (list[tuple]): indices for the matrix to be modified
            corr (list[int]): correlations value

        Returns:
            np.ndarray: A correlation matrix

        """
        for i in range(len(indices)):
            index = indices[i]
            self.mat[index] = corr[i]
            index = index[1], index[0]
            self.mat[index] = corr[i]
        self.make_corr_mat()

        return self.mat
