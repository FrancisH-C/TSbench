"""Correlation matrix wrapper for numpy array."""
import numpy as np
from numpy.random import Generator
from randomgen import Xoshiro256


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

    def __init__(
        self,
        mat: np.ndarray = None,
        rg: Generator = None,
        dim: int = None,
        method: str = "random",
        **method_const
    ):
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
        if mat is not None:
            dim = np.size(mat, 0)

        if rg is None :
            rg = Generator(Xoshiro256())
        self.rg = rg

        self.dim = dim
        self.mat = mat

        self.make_corr_mat = self.normalize_correlation_matrix
        self.method = self.uniform_correlation

        # If needed and you can use `method` to generate the correlation matrix
        if self.mat is None and self.dim is not None:
            self.mat = self.set_mat()

        # If user define, make sure it is a correlation matrices
        # if not mat is None and not self.is_corr_mat():
        #    self.make_corr_mat = self.normalize_correlation_matrix

    def set_dim(self, dim: int) -> np.ndarray:
        """Set dimension for correlation matrix.

        Args:
            dim (int): new dimension to set.
        """
        self.dim = dim
        return self.mat

    def set_method(self, method, **method_const):
        """Set method to generate new correlation matrix.

        Args:
            method (int): new method to use.
        """
        self.method = lambda **method_var: getattr(self, method)(
            **method_const, **method_var
        )
        return self.method

    def set_mat(self, **method_var):
        """Set new correlation matrix using `self.mat`.

        Args:
            **kwargs: keyword arguments used in `self.method`.
        """
        self.mat = self.method(**method_var)
        return self.mat

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

    def mat_from_distribution(self, distribution) -> np.ndarray:
        """Generate a correlation matrix from a distribution function."""
        mat = np.eye(self.dim)
        for i in range(np.size(mat, 0)):
            for j in range(np.size(mat, 1)):
                mat[i, j] = distribution()

        for i in range(np.size(mat, 0)):
            mat[i, i] = 1
        self.mat = mat
        return mat

    def uniform_correlation(self, bias: str = "neutral") -> np.ndarray:
        """Generate a correlation matrix.

        Using Uniform distribution for off-diagonal inputs.

        Args:
            bias (str, optional): Possible value : {"neutral" , "positive" , "negative"}
                - 'Neutral' is uncorrelated
                - 'Positive' is positively correlated
                - 'Negative' is negatively correlated.
                Default is neutral.

        Returns:
            np.ndarray: A random correlation matrix

        """
        if bias == "positive":
            self.mat = self.mat_from_distribution(lambda : self.rg.uniform(0, 1))
        elif bias == "neutral":
            self.mat = self.mat_from_distribution(lambda : self.rg.uniform(-1, 1))
        elif bias == "negative":
            self.mat = self.mat_from_distribution(lambda : self.rg.uniform(-1, 0))
        else:
            raise ValueError('`corr` is either "neutral", "positive" or "negative"')

        return self.mat

    def const(self, corr: int = 0.5) -> np.ndarray:
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

    def specific_corr(self, indices: list[tuple], corr: list[int]) -> np.ndarray:
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
