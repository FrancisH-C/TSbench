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
class Stage:
    """A generate / train / forecast stage.

    Args:
        models: Model instances to run in this stage.
        params: Keyword args passed to ``model.generate``/``train``/
            ``forecast`` (e.g. ``{"N": 100}`` or ``{"T": 10}``).
        input_loaders_params: Extra kwargs for ``input_loader.get_df``.
        id_wise: Generate once per ID (generation stage only).
        rolling_window: Rolling-window config (training stage only); see
            ``Experiment._rolling_window_train_forecast``.

    """

    models: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    input_loaders_params: dict = field(default_factory=dict)
    id_wise: bool = False
    rolling_window: Optional[dict] = None

    def __post_init__(self) -> None:
        if not isinstance(self.models, list):
            raise ValueError("Stage.models must be a list of models.")
        if not isinstance(self.params, dict):
            raise ValueError("Stage.params must be a dict.")


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
