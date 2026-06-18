import pandas as pd

from TSbench.TSdata.TSloader import LoaderTSdf, LoadersProcess
from TSbench.experiment.config import ExperimentConfig
import numpy as np


class Experiment:
    """Benchmarking pipeline.

    Stages: initialize, preprocess, generate, train, forecast, output.
    Configured via a config object whose attributes map to pipeline stages.

    """

    def __init__(self, config):
        """Initialize Experiment from a configuration.

        Args:
            config: Either an :class:`ExperimentConfig` (typed dataclass)
                or a legacy object exposing ``general``, ``initialize``,
                ``pre_process``, ``run_process``, ``generate``, ``train``,
                ``forecast`` and ``output`` dict attributes.

        """
        if isinstance(config, ExperimentConfig):
            config = config.to_dicts()
        self.set_general(config.general)
        self.set_initialize(config.initialize)
        self.set_pre_process(config.pre_process)
        self.set_run_process(config.run_process)
        self.set_generate(config.generate)
        self.set_train(config.train)
        self.set_forecast(config.forecast)
        self.set_output(config.output)

    def set_general(self, general):
        """Set the general configuration (paths, n_jobs, datatype, etc.).

        Args:
            general (dict): General experiment parameters.

        """
        self.general = general
        self._configure_device()

    def _configure_device(self):
        """Apply device configuration from ``self.general``."""
        import os

        device = self.general.get("device")
        if device is None:
            return
        device = device.lower().strip()
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif device.startswith("gpu"):
            parts = device.split(":")
            gpu_id = parts[1] if len(parts) > 1 else "0"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    def set_initialize(self, initialize):
        """Set the initialization stage configuration.

        Args:
            initialize (dict): Must contain a ``'function'`` key with a
                callable that sets up the dataset.

        """
        self.initialize = initialize

    def set_pre_process(self, pre_process):
        """Set the pre-processing stage configuration.

        Builds a ``LoadersProcess`` from the config if one is not provided
        under the ``'process'`` key.

        Args:
            pre_process (dict): Pre-processing parameters.

        """
        self.pre_process = pre_process

        # define process from other inputs
        if "process" not in self.pre_process:
            if "output_loader" not in self.pre_process:
                self.pre_process["output_loader"] = LoaderTSdf(
                    datatype=self.general["datatype"], path=self.general["data_path"]
                )
            if "process_df" not in self.pre_process:
                self.pre_process["process_df"] = None

            if "process_split" not in self.pre_process:
                self.pre_process["process_split"] = None

            pre_process_process = LoadersProcess(
                data_path=self.general["data_path"],
                datatype=self.general["datatype"],
                output_loader=self.pre_process["output_loader"],
                n_jobs=self.general["n_jobs"],
                n_input_loaders=self.general["n_input_loaders"],
                process_df=self.pre_process["process_df"],
                process_split=self.pre_process["process_split"],
            )
            self.pre_process["process"] = pre_process_process

    def set_run_process(self, run_process):
        """Set the main run-process stage configuration.

        Builds a ``LoadersProcess`` from the config if one is not provided
        under the ``'process'`` key.

        Args:
            run_process (dict): Run-process parameters.

        """
        self.run_process = run_process

        # define process from other inputs
        if "process" not in self.run_process:
            if "output_loader" not in self.run_process:
                self.run_process["output_loader"] = LoaderTSdf(
                    datatype=self.general["datatype"], path=self.general["data_path"]
                )
            if "process_df" not in self.run_process:
                self.run_process["process_df"] = None

            if "process_split" not in self.run_process:
                self.run_process["process_split"] = None

            run_process_process = LoadersProcess(
                data_path=self.general["data_path"],
                datatype=self.general["datatype"],
                output_loader=self.run_process["output_loader"],
                n_jobs=self.general["n_jobs"],
                n_input_loaders=self.general["n_input_loaders"],
                process_df=self.run_process["process_df"],
                process_split=self.run_process["process_split"],
            )
            self.run_process["process"] = run_process_process

    def set_generate(self, generate):
        """Set the generation stage configuration.

        Args:
            generate (dict): Must contain ``'models'`` and ``'params'`` keys.

        """
        self.generate = generate
        if "input_loaders_params" not in self.generate:
            self.generate["input_loaders_params"] = {}
        if "ID-wise" not in self.generate:
            self.generate["ID-wise"] = False

    def set_train(self, train):
        """Set the training stage configuration.

        Args:
            train (dict): Must contain ``'models'`` and optionally ``'params'``.

        """
        self.train = train
        if "params" not in self.train:
            self.train["params"] = {}
        if "input_loaders_params" not in self.train:
            self.train["input_loaders_params"] = {}
        if "rolling_window" not in self.train:
            self.train["rolling_window"] = None

    def set_forecast(self, forecast):
        """Set the forecasting stage configuration.

        Args:
            forecast (dict): Must contain ``'models'`` and ``'params'`` keys.

        """
        self.forecast = forecast
        if "input_loaders_params" not in self.forecast:
            self.forecast["input_loaders_params"] = {}

    def set_output(self, output):
        """Set the output stage configuration.

        Builds a ``LoadersProcess`` from the config if one is not provided
        under the ``'process'`` key.

        If a ``"metrics"`` key is present, it should be a list of callables
        with signature ``(y_true, y_pred) -> float``. These are applied
        automatically during the output stage to each forecast feature.

        Args:
            output (dict): Output parameters.

        """
        self.output = output
        if "metrics" not in self.output:
            self.output["metrics"] = None

        # define process from other inputs
        if "process" not in self.output:
            if "output_loader" not in self.output:
                self.output["output_loader"] = LoaderTSdf(
                    datatype=self.general["datatype"], path=self.general["output_path"]
                )
            if "process_df" not in self.output:
                self.output["process_df"] = None

            if "process_split" not in self.output:
                self.output["process_split"] = None

            output_process = LoadersProcess(
                data_path=self.general["data_path"],
                datatype=self.general["datatype"],
                output_loader=self.output["output_loader"],
                n_jobs=self.general["n_jobs"],
                n_input_loaders=self.general["n_input_loaders"],
                process_df=self.output["process_df"],
                process_split=self.output["process_split"],
            )
            self.output["process"] = output_process

    def configure_run_models(self, generate, train, forecast, write=True):
        """Build a closure that runs generate/train/forecast on a loader.

        The returned closure matches the ``process_split`` signature expected
        by ``LoadersProcess``: it takes a single ``input_loader`` argument.
        The output loader is accessed from ``self.run_process``.

        If ``self.train["rolling_window"]`` is set, training uses a rolling
        window over IDs (days) instead of training independently per ID.
        See ``_rolling_window_train`` for details.

        Args:
            generate (bool): Whether to run the generation step.
            train (bool): Whether to run the training step.
            forecast (bool): Whether to run the forecasting step.
            write (bool): Whether to write results to disk.

        Returns:
            Callable: A function ``(input_loader) -> None``.

        """
        output_loader = self.run_process["process"].output_loader

        def run_models(input_loader):
            if generate:
                for model in self.generate["models"]:
                    if self.generate["ID-wise"]:
                        for ID in input_loader.get_IDs():
                            df = input_loader.get_df(
                                IDs=ID, **self.generate["input_loaders_params"]
                            )
                            model.set_data(df)
                            model.generate(**self.generate["params"])
                            model.register_data(input_loader)
                            model.register_data(output_loader)
                    else:
                        df = input_loader.get_df(
                            **self.generate["input_loaders_params"]
                        )
                        model.set_data(df)
                        model.generate(**self.generate["params"])
                        model.register_data(input_loader)
                        model.register_data(output_loader)

            rolling_window = self.train.get("rolling_window")

            if rolling_window is not None and (train or forecast):
                self._rolling_window_train_forecast(
                    input_loader, output_loader, rolling_window, train, forecast
                )
            else:
                # Standard per-ID train/forecast
                for ID in input_loader.get_IDs():
                    if train:
                        for model in self.train["models"]:
                            df = input_loader.get_df(
                                IDs=ID, **self.train["input_loaders_params"]
                            )
                            model.set_data(df)
                            model.train(**self.train["params"])
                    if forecast:
                        for model in self.forecast["models"]:
                            df = input_loader.get_df(
                                IDs=ID, **self.forecast["input_loaders_params"]
                            )
                            model.set_data(df)
                            model.forecast(**self.forecast["params"])
                            model.register_data(
                                output_loader,
                                append_to_feature=str(model),
                                ID=ID,
                            )
            if write:
                output_loader.write()

        return run_models

    def _rolling_window_train_forecast(
        self, input_loader, output_loader, rolling_window, train, forecast
    ):
        """Execute rolling-window train/val/test across IDs within a split.

        Slides a window of size ``train_size + val_size + test_size`` across
        the sorted list of IDs (typically trading days). For each window
        position:

        - The first ``train_size`` IDs are used for training.
        - The next ``val_size`` IDs are used for validation.
        - The last ``test_size`` IDs are used for testing (forecast).

        Models that support validation data (via ``set_validation_data``)
        will receive it automatically. Models are rebuilt at the start of
        each split via ``build_model()`` if ``reset_per_split`` is True
        (default).

        Args:
            input_loader: The input loader for the current split (stock).
            output_loader: The output loader for writing forecasts.
            rolling_window (dict): Rolling window configuration with keys:

                - ``train_size`` (int): Number of IDs for training. Default 1.
                - ``val_size`` (int): Number of IDs for validation. Default 1.
                - ``test_size`` (int): Number of IDs for testing. Default 1.
                - ``step_size`` (int): Step size for sliding. Default 1.
                - ``min_rows`` (int): Skip IDs with fewer rows. Default 0.
                - ``reset_per_split`` (bool): Rebuild model weights at the
                  start of each split (stock). Default True.

            train (bool): Whether to run training.
            forecast (bool): Whether to run forecasting.

        """
        train_size = rolling_window.get("train_size", 1)
        val_size = rolling_window.get("val_size", 1)
        test_size = rolling_window.get("test_size", 1)
        step_size = rolling_window.get("step_size", 1)
        min_rows = rolling_window.get("min_rows", 0)
        reset_per_split = rolling_window.get("reset_per_split", True)

        window_size = train_size + val_size + test_size
        IDs = input_loader.get_IDs()

        if len(IDs) < window_size:
            return

        # Reset models at the start of each split (stock)
        if reset_per_split:
            for model in self.train.get("models", []):
                if hasattr(model, "build_model"):
                    model.model = model.build_model()

        for start in range(0, len(IDs) - window_size + 1, step_size):
            train_IDs = IDs[start : start + train_size]
            val_IDs = IDs[start + train_size : start + train_size + val_size]
            test_IDs = IDs[start + train_size + val_size : start + window_size]

            if train:
                for model in self.train["models"]:
                    # Collect training data from all train IDs
                    train_dfs = []
                    skip = False
                    for ID in train_IDs:
                        df = input_loader.get_df(
                            IDs=ID, **self.train["input_loaders_params"]
                        )
                        if min_rows > 0 and df.shape[0] < min_rows:
                            skip = True
                            break
                        train_dfs.append(df)
                    if skip:
                        continue

                    # Collect validation data from all val IDs
                    val_dfs = []
                    for ID in val_IDs:
                        df = input_loader.get_df(
                            IDs=ID, **self.train["input_loaders_params"]
                        )
                        if min_rows > 0 and df.shape[0] < min_rows:
                            skip = True
                            break
                        val_dfs.append(df)
                    if skip:
                        continue

                    # Check test data availability
                    for ID in test_IDs:
                        df = input_loader.get_df(
                            IDs=ID, **self.train["input_loaders_params"]
                        )
                        if min_rows > 0 and df.shape[0] < min_rows:
                            skip = True
                            break
                    if skip:
                        continue

                    # Set training data (concatenate if multiple IDs)
                    if len(train_dfs) == 1:
                        model.set_data(train_dfs[0])
                    else:
                        model.set_data(pd.concat(train_dfs))

                    # Set validation data if model supports it
                    if hasattr(model, "set_validation_data") and val_dfs:
                        if len(val_dfs) == 1:
                            model.set_validation_data(val_dfs[0])
                        else:
                            model.set_validation_data(pd.concat(val_dfs))

                    model.train(**self.train["params"])

            if forecast:
                for model in self.forecast["models"]:
                    for ID in test_IDs:
                        df = input_loader.get_df(
                            IDs=ID, **self.forecast["input_loaders_params"]
                        )
                        if min_rows > 0 and df.shape[0] < min_rows:
                            continue
                        model.set_data(df)
                        model.forecast(**self.forecast["params"])
                        model.register_data(
                            output_loader,
                            append_to_feature=str(model),
                            ID=ID,
                        )

    def _resolve_stages(
        self, initialize, pre_process, generate, train, forecast, output
    ):
        """Resolve which stages to run, auto-detecting when no flag is set.

        If every flag is ``None``, run all stages that have work configured
        (init function present, a pre-process transform, configured models,
        or output). If any flag is explicitly set, the unspecified flags
        default to ``False``.

        Returns:
            dict: Mapping stage name to bool.

        """
        raw = {
            "initialize": initialize,
            "pre_process": pre_process,
            "generate": generate,
            "train": train,
            "forecast": forecast,
            "output": output,
        }
        if any(v is not None for v in raw.values()):
            return {k: bool(v) for k, v in raw.items()}

        return {
            "initialize": self.initialize.get("function") is not None,
            "pre_process": self.pre_process.get("process_df") is not None
            or self.pre_process.get("process_split") is not None,
            "generate": bool(self.generate.get("models")),
            "train": bool(self.train.get("models")),
            "forecast": bool(self.forecast.get("models")),
            "output": True,
        }

    def run(
        self,
        initialize=None,
        pre_process=None,
        generate=None,
        train=None,
        forecast=None,
        output=None,
    ):
        """Execute the experiment pipeline.

        With no arguments, every configured stage runs (auto-detected). Pass
        ``True``/``False`` to any flag to select a subset; the other
        unspecified flags then default to ``False``.

        Args:
            initialize (bool, optional): Run the initialization function.
            pre_process (bool, optional): Run the pre-processing stage.
            generate (bool, optional): Run model generation.
            train (bool, optional): Run model training.
            forecast (bool, optional): Run model forecasting.
            output (bool, optional): Run the output/post-processing stage.

        """
        stages = self._resolve_stages(
            initialize, pre_process, generate, train, forecast, output
        )

        if stages["initialize"]:
            self.initialize["function"]()

        if stages["pre_process"]:
            self.pre_process["process"].run_process(write=True)

        if stages["generate"] or stages["train"] or stages["forecast"]:
            run_models = self.configure_run_models(
                stages["generate"], stages["train"], stages["forecast"], write=True
            )
            self.run_process["process"].process_split = run_models
            self.run_process["process"].run_process(write=True)

        if stages["output"]:
            self.output["process"].reload()
            self.output["process"].run_process(write=True)
            self._results = self._evaluate_metrics()

    def compute_metrics(self, y_true, y_pred):
        """Evaluate all configured metrics on the given arrays.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            dict: Mapping from metric name to computed value, or empty
            dict if no metrics are configured.

        """
        metrics_fns = self.output.get("metrics")
        if not metrics_fns:
            return {}
        results = {}
        for fn in metrics_fns:
            name = getattr(fn, "__name__", str(fn))
            results[name] = fn(y_true, y_pred)
        return results

    def _evaluate_metrics(self):
        """Compute configured metrics on forecast results.

        For each ID in the output loader, compares the base feature
        (e.g. ``"returns"``) against each forecast feature (e.g.
        ``"returns_ARMA"``). Results are keyed by ``(ID, model, metric)``.

        Returns:
            dict: Nested dict ``{ID: {model: {metric_name: value}}}``,
            or empty dict if no metrics are configured.

        """
        metrics_fns = self.output.get("metrics")
        if not metrics_fns:
            return {}

        output_loader = self.output.get("output_loader")
        if output_loader is None:
            output_loader = self.output.get("process", {})
            if hasattr(output_loader, "output_loader"):
                output_loader = output_loader.output_loader
            else:
                return {}

        results = {}
        try:
            df = output_loader.get_df()
        except Exception:
            return {}

        if df is None or df.empty:
            return {}

        features = df.columns.tolist() if hasattr(df, "columns") else []
        base_features = [f for f in features if "_" not in f]
        forecast_features = [f for f in features if "_" in f]

        for ID in output_loader.get_IDs():
            results[ID] = {}
            try:
                id_df = output_loader.get_df(IDs=ID)
            except Exception:
                continue
            for ff in forecast_features:
                parts = ff.split("_", 1)
                base = parts[0]
                model_name = parts[1] if len(parts) > 1 else ff
                if base not in base_features:
                    continue
                y_true = id_df[base].to_numpy()
                y_pred = id_df[ff].to_numpy()
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                if mask.sum() == 0:
                    continue
                metric_results = self.compute_metrics(y_true[mask], y_pred[mask])
                if metric_results:
                    results[ID][model_name] = metric_results

        return results

    @property
    def results(self):
        """Metric results from the last ``run(output=True)`` call.

        Returns:
            dict: ``{ID: {model: {metric_name: value}}}``, or empty dict.

        """
        return getattr(self, "_results", {})

    def get_output_loader(self):
        """Return the output loader containing experiment results."""
        return self.output["output_loader"]
