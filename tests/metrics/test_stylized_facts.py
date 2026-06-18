"""Tests for stylized facts metrics."""

import numpy as np
from numpy.random import PCG64, Generator

from TSbench.metrics import excess_kurtosis, volatility_clustering, ljung_box


class TestExcessKurtosis:
    def test_gaussian_near_zero(self):
        rg = Generator(PCG64(42))
        x = rg.standard_normal(10000)
        # Gaussian excess kurtosis ~ 0
        assert abs(excess_kurtosis(x)) < 0.2

    def test_heavy_tails_positive(self):
        rg = Generator(PCG64(42))
        x = rg.standard_t(df=3, size=10000)
        # Student-t with df=3 finite samples give large positive values
        assert excess_kurtosis(x) > 1.0

    def test_uniform_negative(self):
        rg = Generator(PCG64(42))
        x = rg.uniform(-1, 1, size=10000)
        # Uniform has excess kurtosis = -1.2
        assert excess_kurtosis(x) < 0


class TestVolatilityClustering:
    def test_output_shape(self):
        rg = Generator(PCG64(42))
        x = rg.standard_normal(100)
        acf = volatility_clustering(x, max_lag=5)
        assert acf.shape == (5,)

    def test_constant_returns_zeros(self):
        x = np.ones(100)
        acf = volatility_clustering(x, max_lag=5)
        np.testing.assert_array_equal(acf, np.zeros(5))

    def test_garch_like_clustering(self):
        # Simulate a simple GARCH(1,1)-like process with volatility clustering
        rg = Generator(PCG64(42))
        n = 5000
        x = np.zeros(n)
        sigma2 = np.zeros(n)
        sigma2[0] = 0.01
        for t in range(1, n):
            sigma2[t] = 0.01 + 0.1 * x[t - 1] ** 2 + 0.85 * sigma2[t - 1]
            x[t] = np.sqrt(sigma2[t]) * rg.standard_normal()
        acf = volatility_clustering(x, max_lag=5)
        # GARCH process should show positive autocorrelation in squared returns
        assert acf[0] > 0.05


class TestLjungBox:
    def test_iid_high_pvalue(self):
        rg = Generator(PCG64(42))
        x = rg.standard_normal(1000)
        _, p = ljung_box(x, max_lag=10)
        # iid noise should not reject null of no autocorrelation
        assert p > 0.01

    def test_autocorrelated_low_pvalue(self):
        rg = Generator(PCG64(42))
        n = 1000
        x = np.zeros(n)
        x[0] = rg.standard_normal()
        for t in range(1, n):
            x[t] = 0.9 * x[t - 1] + rg.standard_normal()
        _, p = ljung_box(x, max_lag=10)
        # Strongly autocorrelated series should reject
        assert p < 0.01

    def test_short_series(self):
        x = np.array([1.0, 2.0, 3.0])
        Q, p = ljung_box(x, max_lag=10)
        assert Q == 0.0
        assert p == 1.0
