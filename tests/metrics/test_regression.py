"""Tests for regression metrics."""

import numpy as np
import pytest

from TSbench.metrics import mae, mse, rmse, mape, smape, r2


class TestMAE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert mae(y_true, y_pred) == pytest.approx(0.5)

    def test_symmetric(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([3.0, 4.0])
        assert mae(y_true, y_pred) == mae(y_pred, y_true)


class TestMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mse(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mse(y_true, y_pred) == pytest.approx(1.0)


class TestRMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_is_sqrt_of_mse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(mse(y_true, y_pred)))

    def test_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # MSE = (9+16)/2 = 12.5, RMSE = sqrt(12.5)
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))


class TestMAPE:
    def test_known_value(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        # |10/100| + |10/200| = 0.1 + 0.05 = 0.15 / 2 = 0.075
        assert mape(y_true, y_pred) == pytest.approx(0.075)


class TestSMAPE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert smape(y, y) == pytest.approx(0.0)

    def test_both_zero(self):
        y = np.array([0.0, 0.0])
        assert smape(y, y) == pytest.approx(0.0)


class TestR2:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2(y, y) == pytest.approx(1.0)

    def test_constant_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])  # mean prediction
        assert r2(y_true, y_pred) == pytest.approx(0.0)

    def test_constant_true(self):
        y_true = np.array([2.0, 2.0, 2.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert r2(y_true, y_pred) == 0.0
