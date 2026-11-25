#!/usr/bin/env python3
"""A user-defined deep-learning forecaster for TSbench.

Defines ``MLPForecaster``, a small feed-forward network that forecasts the next
value from a sliding window of past observations. It subclasses
:class:`KerasModel` and implements only the two abstract hooks — ``build_model``
and ``preprocess``; training, the iterated one-step-ahead ``forecast(T)``, and
device handling are inherited, so the model is a first-class citizen of the
generate-and-benchmark pipeline.

This module is the *reusable artifact*: import ``MLPForecaster`` into any experiment
script. ``benchmark.py`` does exactly that. Requires the ``keras`` extra::

    python -m pip install -e .[keras]
"""

from __future__ import annotations

import numpy as np
from tensorflow import keras

from TSbench.TSmodels import KerasModel


class MLPForecaster(KerasModel):
    """A feed-forward next-step forecaster.

    Forecasts ``y_t`` from the previous ``window`` observations. Only the two
    abstract hooks are implemented; ``train`` and ``forecast`` (iterated
    one-step-ahead) and device handling come from :class:`KerasModel`.
    """

    def __init__(self, window: int = 8, units: int = 32, **model_args) -> None:
        # ``window``/``units`` must be set before ``super().__init__`` runs,
        # because ``KerasModel.__init__`` calls ``build_model()`` at the end.
        self.window = window
        self.units = units
        super().__init__(**model_args)

    def build_model(self):
        """Build and compile the Keras network (abstract hook)."""
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.window,)),
                keras.layers.Dense(self.units, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def preprocess(self, df, fit_scaler=False):
        """Turn the loaded frame into sliding-window ``(X, y)`` pairs.

        Reads only the feature column(s) (not the index), so it works both for
        training and for the recursive ``forecast`` loop.
        """
        del fit_scaler  # this minimal example uses no feature scaling
        series = df.to_numpy(dtype="float32").reshape(-1)
        X = np.stack(
            [series[i : i + self.window] for i in range(len(series) - self.window)]
        )
        y = series[self.window :]
        return X, y
