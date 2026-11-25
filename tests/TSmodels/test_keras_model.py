"""Contract tests for the abstract ``KerasModel`` base.

``keras_model`` imports ``keras`` at module load, so these tests require the
optional ``keras`` package and are skipped without it (``importorskip``).
Construction/device wiring is exercised with a stub subclass whose
``build_model`` returns a plain sentinel (no real backend touched); a small
real Keras model checks that ``forecast(T)`` honors the ``T``-step contract.
The full ``Experiment``-based smoke lives in
``examples/deep_learning/benchmark.py`` (forecaster defined in ``models.py``).
"""

import os

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from TSbench import TSmodels
from TSbench.TSmodels.models import ForecastingModel

keras = pytest.importorskip("keras")  # KerasModel requires the optional keras extra

KerasModel = TSmodels.KerasModel

_SENTINEL = object()


class _StubKeras(KerasModel):
    """Concrete KerasModel whose build_model needs no real backend."""

    def build_model(self):
        return _SENTINEL

    def preprocess(self, df, fit_scaler=False):
        del df, fit_scaler  # interface-only in this stub
        return np.zeros((1, 1)), np.zeros((1,))


class _TinyMLP(KerasModel):
    """A real (tiny) windowed next-step Keras model for the forecast test."""

    def __init__(self, window=3, **model_args):
        self.window = window
        super().__init__(**model_args)

    def build_model(self):
        model = keras.Sequential(
            [keras.layers.Input(shape=(self.window,)), keras.layers.Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def preprocess(self, df, fit_scaler=False):
        del fit_scaler
        series = df.to_numpy(dtype="float32").reshape(-1)
        X = np.stack(
            [series[i : i + self.window] for i in range(len(series) - self.window)]
        )
        return X, series[self.window :]


def test_kerasmodel_exported_and_subclass():
    """KerasModel is re-exported from TSmodels and is a ForecastingModel."""
    assert TSmodels.KerasModel is KerasModel
    assert issubclass(KerasModel, ForecastingModel)


def test_build_model_and_preprocess_are_abstract():
    """Subclasses must override build_model/preprocess (marked abstract)."""
    assert KerasModel.build_model.__isabstractmethod__
    assert KerasModel.preprocess.__isabstractmethod__


def test_stub_subclass_constructs():
    """A concrete subclass builds, defaulting feature_label to ["prediction"]."""
    model = _StubKeras(rg=Generator(PCG64(0)))
    assert model.model is _SENTINEL
    assert list(model.feature_label) == ["prediction"]


def test_configure_device_cpu(monkeypatch):
    """device="cpu" disables the GPU via CUDA_VISIBLE_DEVICES."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _StubKeras(rg=Generator(PCG64(0)), device="cpu")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"


def test_configure_device_none_leaves_env_untouched(monkeypatch):
    """device=None makes no change to CUDA_VISIBLE_DEVICES."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _StubKeras(rg=Generator(PCG64(0)), device=None)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_forecast_honors_T(monkeypatch):
    """forecast(T) returns exactly T values (iterated one-step), like every model."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    model = _TinyMLP(
        window=3, device="cpu", rg=Generator(PCG64(0)), feature_label=["value"]
    )
    model.set_data(data=np.arange(10, dtype=float).reshape(-1, 1))  # timestamps 0..9
    out = model.forecast(T=5)
    # Exactly T forecasts, stamped on the held-out horizon (after the input).
    assert len(out) == 5
    assert np.isfinite(out.to_numpy()).all()
