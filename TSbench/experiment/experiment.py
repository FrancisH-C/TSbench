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

        If ``self.train.rolling`` is set, training/forecasting use a sliding
        window along one index axis (ID or timestamp) instead of training
        independently per ID. See ``_rolling_window`` for details.

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

            rolling = self.train.rolling

            if rolling and (train or forecast):
                self._rolling_window(
                    input_loader, output_loader, rolling, 0, train, forecast
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

    def _axis_values(self, input_loader, axis, unit=None, scope_ids=None):
        """Sorted unique values along the rolling ``axis``.

        For ``axis="ID"`` returns the IDs (restricted to ``scope_ids`` when
        given); for ``axis="timestamp"`` returns the timestamps of the single ID
        ``unit``.
        """
        if axis == "ID":
            ids = list(input_loader.get_IDs())
            if scope_ids is not None:
                scope = set(scope_ids)
                ids = [i for i in ids if i in scope]
            return ids
        return list(input_loader.get_timestamp(IDs=np.array([unit]), unique=True))

    def _iter_windows(self, values, rolling):
        """Yield ``(train_vals, val_vals, test_vals)`` for each window position.

        Honors fixed vs expanding training windows (see :class:`RollingWindow`):
        the test slice slides by ``step_size``; the train slice is the
        ``train_size`` values before the validation slice (fixed) or grows from
        the start capped at ``max_train_size`` (expanding), starting once at
        least ``min_window_size`` (default ``train_size``) values are available.
        """
        n = len(values)
        train_size = rolling.train_size
        val_size = rolling.val_size
        test_size = rolling.test_size
        min_window = rolling.min_window_size or train_size

        first = (min_window if rolling.expanding else train_size) + val_size
        for test_start in range(first, n - test_size + 1, rolling.step_size):
            test_vals = values[test_start : test_start + test_size]
            val_start = test_start - val_size
            val_vals = values[val_start:test_start]
            if rolling.expanding:
                if rolling.max_train_size is None:
                    train_lo = 0
                else:
                    train_lo = max(0, val_start - rolling.max_train_size)
            else:
                train_lo = val_start - train_size
            if train_lo < 0:
                continue
            train_vals = values[train_lo:val_start]
            if len(train_vals) < min_window:
                continue
            yield train_vals, val_vals, test_vals

    def _rolling_window(
        self, input_loader, output_loader, levels, level, train, forecast,
        scope_ids=None,
    ):
        """Nested sliding-window train/forecast along the configured axes.

        ``levels`` is the ``Stage.rolling`` list (outermost first). This
        processes ``levels[level]``: it slides windows along that level's axis,
        (re)fits the train-stage models on each training slice when ``retrain``
        is set, and for each test slice either forecasts directly (the last
        level) or recurses into the next level restricted to that test slice's
        IDs. The model is shared across levels, so a deeper level with
        ``retrain=False`` forecasts with the model an outer level trained (use
        the same model instance in the train and forecast stages).

        See :class:`RollingWindow` for axis and window-sizing semantics.

        Args:
            input_loader: Input loader for the current split.
            output_loader: Output loader for writing forecasts.
            levels (list[RollingWindow]): The nesting (outermost first).
            level (int): Index of the level being processed.
            train (bool): Whether to run training.
            forecast (bool): Whether to run forecasting.
            scope_ids: Restrict to these IDs (set by an outer level); ``None`` =
                all IDs.

        """
        rolling = levels[level]
        axis = rolling.axis
        is_leaf = level == len(levels) - 1

        # Held-fixed grouping: ID axis is a single pass; timestamp axis rolls
        # within each ID (dims kept together).
        if axis == "ID":
            units = [None]
        else:
            units = list(
                input_loader.get_IDs() if scope_ids is None else scope_ids
            )

        for unit in units:
            values = self._axis_values(input_loader, axis, unit, scope_ids)
            for train_vals, val_vals, test_vals in self._iter_windows(values, rolling):
                if train and rolling.retrain:
                    self._fit_window(
                        input_loader, axis, unit, train_vals, val_vals,
                        rolling.min_rows,
                    )
                if forecast:
                    if is_leaf:
                        self._forecast_window(
                            input_loader, output_loader, axis, unit,
                            train_vals, val_vals, test_vals, rolling.min_rows,
                        )
                    else:
                        inner_scope = list(test_vals) if axis == "ID" else [unit]
                        self._rolling_window(
                            input_loader, output_loader, levels, level + 1,
                            train, forecast, scope_ids=inner_scope,
                        )

    def _axis_slice(self, input_loader, axis, unit, vals, params):
        """Loader frame for ``vals`` along ``axis`` (``None`` if ``vals`` empty)."""
        if len(vals) == 0:
            return None
        if axis == "ID":
            return input_loader.get_df(IDs=np.array(list(vals)), **params)
        return input_loader.get_df(
            IDs=unit, timestamps=np.array(list(vals)), **params
        )

    def _fit_window(self, input_loader, axis, unit, train_vals, val_vals, min_rows):
        """Fit the train-stage models on one window's train (+ val) slice."""
        params = self.train.input_loaders_params
        for model in self.train.models:
            train_df = self._axis_slice(input_loader, axis, unit, train_vals, params)
            if train_df is None or (min_rows > 0 and train_df.shape[0] < min_rows):
                continue
            model.set_data(train_df)
            if len(val_vals) and hasattr(model, "set_validation_data"):
                val_df = self._axis_slice(input_loader, axis, unit, val_vals, params)
                if val_df is not None and not (
                    min_rows > 0 and val_df.shape[0] < min_rows
                ):
                    model.set_validation_data(val_df)
            model.train(**self.train.params)

    def _forecast_window(
        self, input_loader, output_loader, axis, unit,
        train_vals, val_vals, test_vals, min_rows,
    ):
        """Forecast one window's test slice and write it to the output loader.

        - ``axis="ID"``: apply each model to each held-out test ID's own series,
          using the forecast stage's ``params``.
        - ``axis="timestamp"``: seed the train+val context, forecast
          ``len(test_vals)`` steps, and place them at the real test timestamps
          (``set_data`` re-stamps from zero, so forecasts are placed explicitly
          to stay aligned with the ground truth for metric pairing).
        """
        if axis == "ID":
            for model in self.forecast.models:
                for ID in test_vals:
                    df = input_loader.get_df(
                        IDs=ID, **self.forecast.input_loaders_params
                    )
                    if min_rows > 0 and df.shape[0] < min_rows:
                        continue
                    model.set_data(df)
                    model.forecast(**self.forecast.params)
                    model.register_data(
                        output_loader, append_to_feature=str(model), ID=ID,
                    )
            return

        context_vals = np.array(list(train_vals) + list(val_vals))
        test_timestamps = np.array(test_vals)
        for model in self.forecast.models:
            model.set_data(input_loader.get_df(IDs=unit, timestamps=context_vals))
            model.forecast(T=len(test_vals))
            predictions = model.get_data(tstype=np.ndarray)
            out_features = np.array(
                [f"{feature}_{model}" for feature in model.feature_label]
            )
            output_loader.add_data(
                data=predictions,
                ID=unit,
                timestamp=test_timestamps,
                dim_label=model.dim_label,
                feature_label=out_features,
                collision="update",
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
