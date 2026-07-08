"""Typed dataclass configuration for :class:`Experiment`.

A user assembles an experiment from small pieces

    from TSbench.experiment import ExperimentConfig, General, Stage, Output

    cfg = ExperimentConfig(
        general=General(data_path="data/exp/", datatype="simulated"),
        generate=Stage(models=[arma], params={"N": 100}),
        train=Stage(models=[arma]),
        forecast=Stage(models=[arma], params={"T": 10}),
        output=Output(metrics=[mae, rmse]),
    )
    Experiment(cfg).run()        # auto-detects configured stages

:class:`Experiment` consumes :class:`ExperimentConfig` directly"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class General:
    """Paths, dataset identity and parallelism options.

    Args:
        data_path: Directory for input/intermediate data.
        output_path: Directory for results. Defaults to ``data_path``.
        datatype: Dataset identifier (subdirectory / logical name).
        n_jobs: Parallel jobs per loader.
        n_input_loaders: Number of input loaders.
        device: ``"cpu"``, ``"gpu"`` or ``"gpu:N"`` (sets
            ``CUDA_VISIBLE_DEVICES``); ``None`` leaves it untouched.

    """

    data_path: str = "data/experiment/"
    output_path: Optional[str] = None
    datatype: str = "simulated"
    n_jobs: int = 1
    n_input_loaders: int = 1
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.data_path, str) or not self.data_path:
            raise ValueError("General.data_path must be a non-empty string.")
        if self.output_path is None:
            self.output_path = self.data_path
        if not isinstance(self.datatype, str) or not self.datatype:
            raise ValueError("General.datatype must be a non-empty string.")
        if self.n_jobs < 1 or self.n_input_loaders < 1:
            raise ValueError("n_jobs and n_input_loaders must be >= 1.")


@dataclass
class RollingWindow:
    """One sliding-window level along an index axis.

    The dataset frame is indexed by ``(ID, timestamp, dim)``. A rolling level
    fixes two of those levels and slides a window along the third (``axis``),
    splitting the ordered values into consecutive train / validation / test
    slices and stepping forward by ``step_size``. Levels are composed as a list
    on ``Stage.rolling`` (see :class:`Stage`); the list order is the nesting
    order.

    The supported axes are ``"ID"`` and ``"timestamp"``:

    - ``axis="ID"``: roll over IDs — train on some series, forecast held-out
      series (cross-series). For a leaf (last) level, ``Experiment`` applies the
      trained model to each test ID's own data using the forecast stage's
      ``params``.
    - ``axis="timestamp"``: walk-forward within each ID — train on early
      timestamps, forecast the next ``test_size`` timestamps. The forecast
      horizon is ``test_size`` (derived from the window, not the forecast
      stage's ``params``); forecasts land on the held-out timestamps so the
      metrics pair against the ground truth.

    ``dim`` is intentionally **not** a rolling axis: rolling on it would mean
    predicting held-out dimensions from observed ones (a cross-sectional
    operation), which is outside the temporal ``forecast(T)`` contract every
    model implements (models fix ``dim`` at construction and forecast all of
    their dimensions jointly, forward in time).

    Training window sizing (``expanding`` toggle):

    - ``expanding=False`` (default): a fixed window of ``train_size`` values
      immediately before the validation/test slices (sliding window).
    - ``expanding=True``: the window grows from the start of the available
      values, capped at ``max_train_size`` (``None`` = unbounded). Forecasting
      starts once at least ``min_window_size`` (``None`` → ``train_size``)
      values are available.

    Args:
        axis: Index level to roll along, ``"ID"`` or ``"timestamp"``.
        train_size: Base/initial number of axis values used for training.
        val_size: Number of axis values used for validation per window
            (``0`` = no validation; passed to ``set_validation_data`` when the
            model supports it).
        test_size: Number of axis values forecast per window.
        step_size: How far the window advances each step.
        retrain: Whether to (re)fit the model at each window. ``False`` freezes
            the model (no fit, no weight rebuild) and only forecasts — used by a
            deeper level to reuse the model an outer level trained.
        expanding: Grow the training window instead of using a fixed size.
        max_train_size: Cap on the training window when ``expanding`` (``None``
            = unbounded); ignored when not expanding.
        min_window_size: Minimum training length before the first forecast
            (``None`` → ``train_size``).
        min_rows: Skip a slice whose row count is below this (0 = no filter).

    """

    axis: str = "ID"
    train_size: int = 1
    val_size: int = 1
    test_size: int = 1
    step_size: int = 1
    retrain: bool = True
    expanding: bool = False
    max_train_size: Optional[int] = None
    min_window_size: Optional[int] = None
    min_rows: int = 0

    def __post_init__(self) -> None:
        if self.axis not in ("ID", "timestamp"):
            raise ValueError(
                "RollingWindow.axis must be 'ID' or 'timestamp'. Rolling on "
                "'dim' would require cross-sectional forecasting (predicting "
                "held-out dimensions from observed ones), which the models do "
                "not provide."
            )
        if min(self.train_size, self.test_size, self.step_size) < 1:
            raise ValueError(
                "RollingWindow train_size/test_size/step_size must be >= 1."
            )
        if self.val_size < 0:
            raise ValueError("RollingWindow val_size must be >= 0.")
        for name in ("max_train_size", "min_window_size"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"RollingWindow.{name}, when set, must be >= 1.")


@dataclass
class Stage:
    """A generate / train / forecast stage.

    Args:
        models: Model instances to run in this stage.
        params: Keyword args passed to ``model.generate``/``train``/
            ``forecast`` (e.g. ``{"N": 100}`` or ``{"T": 10}``).
        input_loaders_params: Extra kwargs for ``input_loader.get_df``.
        id_wise: Generate once per ID (generation stage only).
        rolling: List of :class:`RollingWindow` levels (training stage only).
            **The order of the list is the order of nesting**: the first entry
            is the outermost roll, and each later entry rolls within the test
            units of the entry before it. A single-level roll is a one-element
            list. See ``Experiment._rolling_window``.

    """

    models: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    input_loaders_params: dict = field(default_factory=dict)
    id_wise: bool = False
    rolling: Optional[list[RollingWindow]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.models, list):
            raise ValueError("Stage.models must be a list of models.")
        if not isinstance(self.params, dict):
            raise ValueError("Stage.params must be a dict.")
        if self.rolling is not None:
            if not isinstance(self.rolling, list) or not self.rolling:
                raise ValueError(
                    "Stage.rolling must be a non-empty list of RollingWindow "
                    "(outermost first)."
                )
            if not all(isinstance(r, RollingWindow) for r in self.rolling):
                raise ValueError("Stage.rolling entries must be RollingWindow.")


@dataclass
class Process:
    """Advanced pre-process / run-process stage (escape hatch).

    Leave the stage as ``None`` on :class:`ExperimentConfig` for the
    default behavior. Provide a :class:`Process` to customize the
    per-DataFrame / per-split transforms or the output loader.

    Args:
        process_df: ``(df) -> df`` applied per ID.
        process_split: ``(input_loader) -> None`` applied per split.
        output_loader: Pre-built output loader.

    """

    process_df: Optional[Callable] = None
    process_split: Optional[Callable] = None
    output_loader: Optional[Any] = None


@dataclass
class Output:
    """Output stage configuration.

    Args:
        metrics: Metric callables ``(y_true, y_pred) -> float`` applied to
            each forecast feature; ``None`` disables metric evaluation.
        process_split: Optional advanced per-split transform.
        process_df: Optional advanced per-DataFrame transform.
        output_loader: Optional pre-built output loader.

    """

    metrics: Optional[list[Callable]] = None
    process_split: Optional[Callable] = None
    process_df: Optional[Callable] = None
    output_loader: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.metrics is not None:
            if not isinstance(self.metrics, list) or not all(
                callable(m) for m in self.metrics
            ):
                raise ValueError("Output.metrics must be a list of callables.")


@dataclass
class ExperimentConfig:
    """Typed configuration for an :class:`Experiment`.

    Args:
        general: Paths / dataset / parallelism options.
        generate: Generation stage.
        train: Training stage.
        forecast: Forecasting stage.
        output: Output stage (metrics, advanced transforms).
        initialize: Zero-arg callable that prepares the dataset. When
            ``None`` (default) a function that restarts an empty dataset
            at the configured path is used.
        pre_process: Advanced pre-processing stage (``None`` = default).
        run_process: Advanced run-process stage (``None`` = default).

    """

    general: General = field(default_factory=General)
    generate: Stage = field(default_factory=Stage)
    train: Stage = field(default_factory=Stage)
    forecast: Stage = field(default_factory=Stage)
    output: Output = field(default_factory=Output)
    initialize: Optional[Callable] = None
    pre_process: Optional[Process] = None
    run_process: Optional[Process] = None

    def __post_init__(self) -> None:
        rolling = self.train.rolling
        if (
            rolling is not None
            and any(level.axis == "timestamp" for level in rolling)
            and "T" in self.forecast.params
        ):
            raise ValueError(
                "forecast stage params must not set 'T' when any train.rolling "
                "level rolls on the 'timestamp' axis: the forecast horizon is "
                "derived from that level's test_size, so a forecast 'T' would "
                "be silently ignored."
            )
