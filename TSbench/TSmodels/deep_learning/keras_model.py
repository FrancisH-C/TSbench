"""Keras deep learning model base class for TSbench."""

from __future__ import annotations

import os
from abc import abstractmethod
from typing import Optional, Union

import keras
import numpy as np
import pandas as pd

from TSbench.TSdata.data import AnyData
from TSbench.TSmodels.models import ForecastingModel


class KerasModel(ForecastingModel):
    """Base class for Keras deep learning forecasting models.

    Subclasses must implement ``build_model`` and ``preprocess``.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        callbacks (list): Keras callbacks for training.
        validation_split (float): Fraction of training data used for
            validation when no explicit validation data is set.
        verbose (int): Verbosity for fit/predict.
        device (str or None): Compute device — ``"cpu"``, ``"gpu"``, or
            ``"gpu:N"``. ``None`` uses the system default.
        **model_args: Arguments forwarded to ``ForecastingModel``.

    """

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks: Optional[list] = None,
        validation_split: float = 0.1,
        verbose: int = 0,
        device: Optional[str] = None,
        **model_args,
    ) -> None:
        """Initialize KerasModel."""
        self.epochs = epochs
        self.batch_size = batch_size
        self.callbacks = callbacks if callbacks is not None else []
        self.validation_split = validation_split
        self.verbose = verbose
        self._validation_df = None
        self.device = device
        self._configure_device()
        super().__init__(**model_args)
        self.model = self.build_model()

    def _configure_device(self) -> None:
        """Configure the compute device via ``CUDA_VISIBLE_DEVICES``.

        Interprets the ``device`` attribute:

        - ``"cpu"`` — disables GPU (sets ``CUDA_VISIBLE_DEVICES="-1"``).
        - ``"gpu"`` or ``"gpu:N"`` — restricts to GPU *N* (default 0).
        - ``None`` — no change (use system default).

        """
        if self.device is None:
            return
        device = self.device.lower().strip()
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif device.startswith("gpu"):
            parts = device.split(":")
            gpu_id = parts[1] if len(parts) > 1 else "0"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    def set_feature_label(self, feature_label: Optional[np.ndarray] = None) -> None:
        """Set feature labels.

        Overrides BaseModel to default to ``["prediction"]`` and allow
        any number of feature labels.

        """
        if feature_label is None:
            feature_label = np.array(["prediction"])
        self.feature_label = feature_label

    @abstractmethod
    def build_model(self):
        """Build and return a compiled Keras model.

        Returns:
            keras.Model: A compiled Keras model.

        """
        pass

    @abstractmethod
    def preprocess(
        self, df: pd.DataFrame, fit_scaler: bool = False
    ) -> tuple[Union[list, np.ndarray], np.ndarray]:
        """Convert a DataFrame into model inputs and targets.

        Args:
            df (pd.DataFrame): Input data from a loader.
            fit_scaler (bool): If True, fit any internal scaler on this
                data. If False, use the previously fitted scaler.

        Returns:
            tuple: ``(inputs, targets)`` where ``inputs`` is an array or
                list of arrays, and ``targets`` is a numpy array.

        """
        pass

    def set_validation_data(self, df: pd.DataFrame) -> None:
        """Set validation data for the next ``train()`` call.

        Args:
            df (pd.DataFrame): Validation DataFrame. Pass ``None`` to
                clear and fall back to ``validation_split``.

        """
        self._validation_df = df

    def train(self) -> "KerasModel":
        """Train the Keras model on loaded data.

        Uses validation data set via ``set_validation_data()`` if
        available, otherwise uses ``validation_split``.

        Returns:
            KerasModel: The trained model (self).

        """
        df = self.get_data()
        inputs, targets = self.preprocess(df, fit_scaler=True)

        fit_kwargs = dict(
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self.callbacks,
            verbose=self.verbose,
        )

        if self._validation_df is not None:
            val_inputs, val_targets = self.preprocess(self._validation_df)
            fit_kwargs["validation_data"] = (val_inputs, val_targets)
        else:
            fit_kwargs["validation_split"] = self.validation_split

        self.model.fit(inputs, targets, **fit_kwargs)
        return self

    def forecast(
        self,
        T: int = 1,
        reset_timestamp: bool = False,
        collision: str = "overwrite",
    ) -> AnyData:
        """Forecast ``T`` steps ahead via iterated one-step prediction.

        Honors the ``forecast(T)`` contract shared by every model in the
        repo (so a ``KerasModel`` can be benchmarked through ``Experiment``):
        seed with the most recent window of loaded data, predict the next
        value, append it to the history, and repeat ``T`` times. The ``T``
        predictions are written right after the loaded input (their
        held-out timestamps), like ``ARMA.forecast``.

        This relies on a small contract for subclasses: ``preprocess`` must
        read the **feature columns** (e.g. ``df.to_numpy()`` or by
        ``feature_label``) rather than the frame's index, and ``model.predict``
        on the last window must return the next observation. The loaded
        history must hold at least one full window.

        Args:
            T (int): Number of steps to forecast.
            reset_timestamp (bool): Whether to reset timestamps.
            collision (str): Collision strategy for ``set_data``.

        Returns:
            AnyData: The ``T`` forecast values stored in the loader.

        """
        history = np.asarray(self.get_data(tstype=np.ndarray), dtype=float)
        if history.ndim == 1:
            history = history.reshape(-1, 1)

        series = history.copy()
        predictions = []
        for _ in range(T):
            frame = pd.DataFrame(series, columns=self.feature_label)
            inputs = np.asarray(self.preprocess(frame)[0])
            if len(inputs) == 0:
                raise ValueError(
                    "forecast needs at least one full window of history."
                )
            step = np.asarray(
                self.model.predict(inputs[-1:], verbose=self.verbose)
            ).reshape(1, -1)
            predictions.append(step[0])
            series = np.vstack([series, step])

        return self.set_data(
            data=np.asarray(predictions),
            reset_timestamp=reset_timestamp,
            collision=collision,
        )

    def save_model(self, filename: str) -> None:
        """Save the Keras model.

        For ``.keras`` files, uses Keras native saving. Otherwise
        falls back to the parent pickle/json method.

        Args:
            filename (str): Path to save to.

        """
        _, ext = os.path.splitext(filename)
        if ext == ".keras":
            self.model.save(filename)
        else:
            super().save_model(filename)

    @classmethod
    def load_model(cls, filename: str):
        """Load a Keras model from disk.

        For ``.keras`` files, uses backend-agnostic Keras 3 saving and
        returns the raw Keras model (not the KerasModel wrapper). For
        pickle files, returns the full object.

        Args:
            filename (str): Path to load from.

        """
        _, ext = os.path.splitext(filename)
        if ext == ".keras":
            return keras.saving.load_model(filename)
        else:
            return super().load_model(filename)
