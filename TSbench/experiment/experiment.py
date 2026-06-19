import pandas as pd

from TSbench.TSdata.TSloader import LoaderTSdf, LoadersProcess
from TSbench.experiment.config import ExperimentConfig
import numpy as np


class Experiment:
    """Benchmarking pipeline.

    Stages: initialize, preprocess, generate, train, forecast, output.
    Configured from a typed :class:`ExperimentConfig`; the config's
    dataclasses (``general``/``generate``/``train``/``forecast``/``output``)
    are read directly as the pipeline's stage definitions, while the built
    ``LoadersProcess`` objects (runtime state) live on attributes.

    """

    def __init__(self, config):
        """Initialize Experiment from a configuration.

        Args:
            config (ExperimentConfig): Typed experiment configuration. Its
                stages are stored as attributes (``self.general``,
                ``self.generate``, ...) and the pre-/run-/output
                ``LoadersProcess`` objects are built on attributes
                (``self._pre_process``, ``self._run_process``,
                ``self._output_process``).

        Raises:
            TypeError: If ``config`` is not an :class:`ExperimentConfig`.

        """
        if not isinstance(config, ExperimentConfig):
            raise TypeError(
                f"Experiment requires an ExperimentConfig; got {type(config).__name__}."
            )
        self.config = config
        self.general = config.general
        self.generate = config.generate
        self.train = config.train
        self.forecast = config.forecast
        self.output = config.output
        self.initialize = config.initialize or self._default_initialize()

        self._configure_device()
        self._pre_process = self._build_process(config.pre_process)
        self._run_process = self._build_process(config.run_process)
        self._output_loader, self._output_process = self._build_output_process()

    def _default_initialize(self):
        """Build the default dataset-restart initialize callable."""
        datatype = self.general.datatype
        path = self.general.data_path

        def initialize():
            LoaderTSdf(datatype=datatype, path=path).restart_dataset()

        return initialize

    def _configure_device(self):
        """Apply device configuration from ``self.general``."""
        import os

        device = self.general.device
        if device is None:
            return
        device = device.lower().strip()
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif device.startswith("gpu"):
            parts = device.split(":")
            gpu_id = parts[1] if len(parts) > 1 else "0"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    def _build_process(self, process):
        """Build a ``LoadersProcess`` for a pre-/run-process stage.

        Args:
            process (Process | None): Advanced stage config, or ``None`` for
                the default (no per-DataFrame / per-split transform, a fresh
                output loader at ``data_path``).

        Returns:
            LoadersProcess: Process bound to ``self.general``'s paths and
            parallelism options.

        """
        process_df = process.process_df if process is not None else None
        process_split = process.process_split if process is not None else None
        output_loader = process.output_loader if process is not None else None
        if output_loader is None:
            output_loader = LoaderTSdf(
                datatype=self.general.datatype, path=self.general.data_path
            )
        return LoadersProcess(
            data_path=self.general.data_path,
            datatype=self.general.datatype,
            output_loader=output_loader,
            n_jobs=self.general.n_jobs,
            n_input_loaders=self.general.n_input_loaders,
            process_df=process_df,
            process_split=process_split,
        )

    def _build_output_process(self):
        """Build the output loader and its ``LoadersProcess``.

        The output loader is written to ``output_path`` (which defaults to
        ``data_path``) and is reused by ``_evaluate_metrics`` and
        ``get_output_loader``.

        Returns:
            tuple: ``(output_loader, output_process)``.

        """
        output = self.output
        output_loader = output.output_loader
        if output_loader is None:
            output_loader = LoaderTSdf(
                datatype=self.general.datatype, path=self.general.output_path
            )
        process = LoadersProcess(
            data_path=self.general.data_path,
            datatype=self.general.datatype,
            output_loader=output_loader,
            n_jobs=self.general.n_jobs,
            n_input_loaders=self.general.n_input_loaders,
            process_df=output.process_df,
            process_split=output.process_split,
        )
        return output_loader, process

    def configure_run_models(self, generate, train, forecast, write=True):
        """Build a closure that runs generate/train/forecast on a loader.

        The returned closure matches the ``process_split`` signature expected
        by ``LoadersProcess``: it takes a single ``input_loader`` argument.
        The output loader is the run-process loader (``self._run_process``).

        If ``self.train.rolling_window`` is set, training uses a rolling
        window over IDs (days) instead of training independently per ID.
        See ``_rolling_window_train_forecast`` for details.

        Args:
            generate (bool): Whether to run the generation step.
            train (bool): Whether to run the training step.
            forecast (bool): Whether to run the forecasting step.
            write (bool): Whether to write results to disk.

        Returns:
            Callable: A function ``(input_loader) -> None``.

        """
        output_loader = self._run_process.output_loader

        def run_models(input_loader):
            if generate:
                for model in self.generate.models:
                    if self.generate.id_wise:
                        for ID in input_loader.get_IDs():
                            df = input_loader.get_df(
                                IDs=ID, **self.generate.input_loaders_params
                            )
                            model.set_data(df)
                            model.generate(**self.generate.params)
                            model.register_data(input_loader)
                            model.register_data(output_loader)
                    else:
                        df = input_loader.get_df(**self.generate.input_loaders_params)
                        model.set_data(df)
                        model.generate(**self.generate.params)
                        model.register_data(input_loader)
                        model.register_data(output_loader)

            rolling_window = self.train.rolling_window

            if rolling_window is not None and (train or forecast):
                self._rolling_window_train_forecast(
                    input_loader, output_loader, rolling_window, train, forecast
                )
            else:
                # Standard per-ID train/forecast
                for ID in input_loader.get_IDs():
                    if train:
                        for model in self.train.models:
                            df = input_loader.get_df(
                                IDs=ID, **self.train.input_loaders_params
                            )
                            model.set_data(df)
                            model.train(**self.train.params)
                    if forecast:
                        for model in self.forecast.models:
                            df = input_loader.get_df(
                                IDs=ID, **self.forecast.input_loaders_params
                            )
                            model.set_data(df)
                            model.forecast(**self.forecast.params)
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
            for model in self.train.models:
                if hasattr(model, "build_model"):
                    model.model = model.build_model()

        for start in range(0, len(IDs) - window_size + 1, step_size):
            train_IDs = IDs[start : start + train_size]
            val_IDs = IDs[start + train_size : start + train_size + val_size]
            test_IDs = IDs[start + train_size + val_size : start + window_size]

            if train:
                for model in self.train.models:
                    # Collect training data from all train IDs
                    train_dfs = []
                    skip = False
                    for ID in train_IDs:
                        df = input_loader.get_df(
                            IDs=ID, **self.train.input_loaders_params
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
                            IDs=ID, **self.train.input_loaders_params
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
                            IDs=ID, **self.train.input_loaders_params
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

                    model.train(**self.train.params)

            if forecast:
                for model in self.forecast.models:
                    for ID in test_IDs:
                        df = input_loader.get_df(
                            IDs=ID, **self.forecast.input_loaders_params
                        )
                        if min_rows > 0 and df.shape[0] < min_rows:
                            continue
                        model.set_data(df)
                        model.forecast(**self.forecast.params)
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

        pre = self.config.pre_process
        return {
            "initialize": self.initialize is not None,
            "pre_process": pre is not None
            and (pre.process_df is not None or pre.process_split is not None),
            "generate": bool(self.generate.models),
            "train": bool(self.train.models),
            "forecast": bool(self.forecast.models),
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
            self.initialize()

        if stages["pre_process"]:
            self._pre_process.run_process(write=True)

        if stages["generate"] or stages["train"] or stages["forecast"]:
            run_models = self.configure_run_models(
                stages["generate"], stages["train"], stages["forecast"], write=True
            )
            self._run_process.process_split = run_models
            self._run_process.run_process(write=True)

        if stages["output"]:
            self._output_process.reload()
            self._output_process.run_process(write=True)
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
        metrics_fns = self.output.metrics
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
        metrics_fns = self.output.metrics
        if not metrics_fns:
            return {}

        output_loader = self._output_loader

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
        return self._output_loader
