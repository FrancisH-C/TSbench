"""Tests for the Experiment class.

Covers:
- Basic pipeline (univariate and multivariate models)
- joblib parallelism (n_jobs > 1, n_input_loaders > 1)
- Metrics evaluation
- Rolling window train/forecast
"""

import shutil

import numpy as np
import pytest
from numpy.random import Generator, PCG64

from TSbench.experiment import (
    Experiment,
    ExperimentConfig,
    General,
    Stage,
    RollingWindow,
    Output,
)
from TSbench.metrics import mae, rmse
from TSbench.TSdata.TSloader import LoaderTSdf
from TSbench.TSmodels import Constant, ARMA, VEC_SPD_GARCH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    data_path,
    models_generate,
    models_train,
    models_forecast,
    n_jobs=1,
    n_input_loaders=1,
    rolling=None,
    metrics=None,
    device=None,
    id_wise=False,
):
    """Build an ExperimentConfig for Experiment tests."""
    return ExperimentConfig(
        general=General(
            data_path=data_path,
            n_jobs=n_jobs,
            n_input_loaders=n_input_loaders,
            device=device,
        ),
        generate=Stage(models=models_generate, params={"N": 20}, id_wise=id_wise),
        train=Stage(models=models_train, rolling=rolling),
        forecast=Stage(models=models_forecast, params={"T": 5}),
        output=Output(metrics=metrics),
    )


def _cleanup(path):
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Univariate pipeline
# ---------------------------------------------------------------------------

class TestExperimentUnivariate:
    """Basic univariate pipeline tests."""

    DATA_PATH = "data/test_experiment_uni/"

    def _models(self, seed=1234):
        feature_label = ["feature"]
        dim_label = ["first"]
        rg_gen = Generator(PCG64(seed))
        rg_train = Generator(PCG64(seed + 1))
        cnst = Constant(
            rg=rg_gen, dim_label=dim_label, feature_label=feature_label
        )
        arma = ARMA(
            lag=1, rg=rg_train, dim_label=dim_label, feature_label=feature_label
        )
        return cnst, arma

    def test_full_pipeline(self):
        """Run init, preprocess, generate, train, forecast and output."""
        cnst, arma = self._models()
        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[cnst, arma],
            models_train=[cnst, arma],
            models_forecast=[arma],
        )
        try:
            exp = Experiment(cfg)
            exp.run(
                initialize=True,
                pre_process=True,
                generate=True,
                train=True,
                forecast=True,
                output=True,
            )
        finally:
            _cleanup(self.DATA_PATH)

    def test_generate_only(self):
        """Run only generation, verify data is written."""
        cnst, _ = self._models()
        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[cnst],
            models_train=[cnst],
            models_forecast=[cnst],
        )
        try:
            exp = Experiment(cfg)
            exp.run(initialize=True, generate=True)
            loader = LoaderTSdf(datatype="simulated", path=self.DATA_PATH)
            df = loader.get_df()
            assert df is not None
            assert len(df) > 0
        finally:
            _cleanup(self.DATA_PATH)


# ---------------------------------------------------------------------------
# Multivariate pipeline
# ---------------------------------------------------------------------------

class TestExperimentMultivariate:
    """Multivariate model pipeline tests."""

    DATA_PATH = "data/test_experiment_multi/"

    def _multivariate_models(self, dim=2, seed=42):
        dim_label = [f"d{i}" for i in range(dim)]
        feature_label = ["feature"]
        rg = Generator(PCG64(seed))
        cnst = Constant(
            dim=dim, rg=rg, dim_label=dim_label, feature_label=feature_label
        )
        return cnst

    def test_multivariate_constant(self):
        """Generate and forecast with a multivariate Constant model."""
        for dim in [2, 3, 5]:
            cnst = self._multivariate_models(dim=dim)
            cfg = _make_config(
                self.DATA_PATH,
                models_generate=[cnst],
                models_train=[cnst],
                models_forecast=[cnst],
            )
            try:
                exp = Experiment(cfg)
                exp.run(
                    initialize=True,
                    generate=True,
                    train=True,
                    forecast=True,
                    output=True,
                )
            finally:
                _cleanup(self.DATA_PATH)

    def test_multivariate_arma(self):
        """Generate and forecast with a multivariate ARMA model."""
        dim = 2
        dim_label = [f"d{i}" for i in range(dim)]
        feature_label = ["feature"]
        rg = Generator(PCG64(99))
        arma = ARMA(
            lag=1, dim=dim, rg=rg, dim_label=dim_label, feature_label=feature_label
        )
        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[arma],
            models_train=[arma],
            models_forecast=[arma],
        )
        try:
            exp = Experiment(cfg)
            exp.run(
                initialize=True,
                generate=True,
                train=True,
                forecast=True,
                output=True,
            )
        finally:
            _cleanup(self.DATA_PATH)

    def test_multivariate_vec_spd_garch(self):
        """Generate with a multivariate VEC-SPD-GARCH model."""
        dim = 2
        dim_label = [f"d{i}" for i in range(dim)]
        rg = Generator(PCG64(55))
        garch = VEC_SPD_GARCH(dim=dim, lag=1, rg=rg, dim_label=dim_label)
        # VEC_SPD_GARCH is GeneratorModel only, use Constant for train/forecast
        cnst = Constant(
            dim=dim,
            rg=Generator(PCG64(56)),
            dim_label=dim_label,
            feature_label=["feature"],
        )
        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[garch],
            models_train=[cnst],
            models_forecast=[cnst],
        )
        try:
            exp = Experiment(cfg)
            exp.run(initialize=True, generate=True)
        finally:
            _cleanup(self.DATA_PATH)


# ---------------------------------------------------------------------------
# Joblib parallelism
# ---------------------------------------------------------------------------

class TestExperimentJoblib:
    """Test that joblib parallelism works correctly."""

    DATA_PATH_SEQ = "data/test_experiment_seq/"
    DATA_PATH_PAR = "data/test_experiment_par/"

    def _run_experiment(self, data_path, n_jobs, n_input_loaders, seed=1234):
        feature_label = ["feature"]
        dim_label = ["first"]
        cnst = Constant(
            rg=Generator(PCG64(seed)),
            dim_label=dim_label,
            feature_label=feature_label,
        )
        arma = ARMA(
            lag=1,
            rg=Generator(PCG64(seed + 1)),
            dim_label=dim_label,
            feature_label=feature_label,
        )
        cfg = _make_config(
            data_path,
            models_generate=[cnst, arma],
            models_train=[cnst, arma],
            models_forecast=[arma],
            n_jobs=n_jobs,
            n_input_loaders=n_input_loaders,
        )
        exp = Experiment(cfg)
        exp.run(
            initialize=True,
            pre_process=True,
            generate=True,
            train=True,
            forecast=True,
            output=True,
        )

    def test_n_jobs_parallel(self):
        """Pipeline with n_jobs=2 completes without error."""
        try:
            self._run_experiment(self.DATA_PATH_PAR, n_jobs=2, n_input_loaders=1)
        finally:
            _cleanup(self.DATA_PATH_PAR)

    def test_sequential_vs_parallel_both_complete(self):
        """Both sequential and parallel runs complete without error."""
        try:
            self._run_experiment(self.DATA_PATH_SEQ, n_jobs=1, n_input_loaders=1)
        finally:
            _cleanup(self.DATA_PATH_SEQ)

        try:
            self._run_experiment(self.DATA_PATH_PAR, n_jobs=2, n_input_loaders=1)
        finally:
            _cleanup(self.DATA_PATH_PAR)


# ---------------------------------------------------------------------------
# Metrics evaluation
# ---------------------------------------------------------------------------

class TestExperimentMetrics:
    """Test metrics computation in the output stage."""

    DATA_PATH = "data/test_experiment_metrics/"

    def test_metrics_computed(self):
        """Metrics are computed when configured."""
        feature_label = ["feature"]
        dim_label = ["first"]
        seed = 1234
        cnst = Constant(
            rg=Generator(PCG64(seed)),
            dim_label=dim_label,
            feature_label=feature_label,
        )

        def mae(y_true, y_pred):
            return np.mean(np.abs(y_true - y_pred))

        def rmse(y_true, y_pred):
            return np.sqrt(np.mean((y_true - y_pred) ** 2))

        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[cnst],
            models_train=[cnst],
            models_forecast=[cnst],
            metrics=[mae, rmse],
        )
        try:
            exp = Experiment(cfg)
            exp.run(
                initialize=True,
                generate=True,
                train=True,
                forecast=True,
                output=True,
            )
            results = exp.results
            # Constant forecasts itself, so metrics should be near zero
            for ID in results:
                for model_name in results[ID]:
                    assert "mae" in results[ID][model_name]
                    assert "rmse" in results[ID][model_name]
                    assert results[ID][model_name]["mae"] >= 0
                    assert results[ID][model_name]["rmse"] >= 0
        finally:
            _cleanup(self.DATA_PATH)

    def test_no_metrics(self):
        """Results are empty when no metrics are configured."""
        feature_label = ["feature"]
        dim_label = ["first"]
        seed = 1234
        cnst = Constant(
            rg=Generator(PCG64(seed)),
            dim_label=dim_label,
            feature_label=feature_label,
        )
        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[cnst],
            models_train=[cnst],
            models_forecast=[cnst],
        )
        try:
            exp = Experiment(cfg)
            exp.run(
                initialize=True,
                generate=True,
                train=True,
                forecast=True,
                output=True,
            )
            assert exp.results == {}
        finally:
            _cleanup(self.DATA_PATH)


# ---------------------------------------------------------------------------
# Rolling window
# ---------------------------------------------------------------------------

class TestExperimentRollingWindow:
    """Test rolling window train/forecast."""

    DATA_PATH = "data/test_experiment_rolling/"

    def _setup_multi_id_data(self, data_path, n_ids=5, N=20):
        """Create a dataset with multiple IDs so rolling window can slide."""
        loader = LoaderTSdf(datatype="simulated", path=data_path)
        loader.restart_dataset()
        for i in range(n_ids):
            cnst = Constant(
                rg=Generator(PCG64(100 + i)),
                dim_label=["first"],
                feature_label=["feature"],
                name=f"ID_{i}",
            )
            cnst.generate(N)
            cnst.register_data(loader, collision="overwrite")
        loader.write()
        return loader

    def test_rolling_window(self):
        """Rolling window train/forecast over multiple IDs."""
        try:
            self._setup_multi_id_data(self.DATA_PATH, n_ids=5, N=20)

            feature_label = ["feature"]
            dim_label = ["first"]
            cnst_train = Constant(
                rg=Generator(PCG64(77)),
                dim_label=dim_label,
                feature_label=feature_label,
            )

            rolling = [
                RollingWindow(
                    axis="ID",
                    train_size=1,
                    val_size=1,
                    test_size=1,
                    step_size=1,
                )
            ]

            cfg = _make_config(
                self.DATA_PATH,
                models_generate=[],
                models_train=[cnst_train],
                models_forecast=[cnst_train],
                rolling=rolling,
            )
            # Don't re-initialize — data already set up
            exp = Experiment(cfg)
            exp.run(train=True, forecast=True)
        finally:
            _cleanup(self.DATA_PATH)

    def test_rolling_window_min_rows(self):
        """Rolling window with min_rows filter."""
        try:
            self._setup_multi_id_data(self.DATA_PATH, n_ids=5, N=10)

            feature_label = ["feature"]
            dim_label = ["first"]
            cnst_train = Constant(
                rg=Generator(PCG64(77)),
                dim_label=dim_label,
                feature_label=feature_label,
            )

            rolling = [
                RollingWindow(
                    axis="ID",
                    train_size=1,
                    val_size=1,
                    test_size=1,
                    step_size=1,
                    min_rows=5,
                )
            ]

            cfg = _make_config(
                self.DATA_PATH,
                models_generate=[],
                models_train=[cnst_train],
                models_forecast=[cnst_train],
                rolling=rolling,
            )
            exp = Experiment(cfg)
            exp.run(train=True, forecast=True)
        finally:
            _cleanup(self.DATA_PATH)

    def test_rolling_window_too_few_ids(self):
        """Rolling window with fewer IDs than window_size returns early."""
        try:
            # Only 2 IDs but window needs 3
            self._setup_multi_id_data(self.DATA_PATH, n_ids=2, N=20)

            feature_label = ["feature"]
            dim_label = ["first"]
            cnst_train = Constant(
                rg=Generator(PCG64(77)),
                dim_label=dim_label,
                feature_label=feature_label,
            )

            rolling = [
                RollingWindow(
                    axis="ID",
                    train_size=1,
                    val_size=1,
                    test_size=1,
                )
            ]

            cfg = _make_config(
                self.DATA_PATH,
                models_generate=[],
                models_train=[cnst_train],
                models_forecast=[cnst_train],
                rolling=rolling,
            )
            exp = Experiment(cfg)
            # Should complete without error (early return)
            exp.run(train=True, forecast=True)
        finally:
            _cleanup(self.DATA_PATH)

    def test_rolling_window_multi_train_ids(self):
        """Rolling window with multiple train IDs concatenates data."""
        try:
            self._setup_multi_id_data(self.DATA_PATH, n_ids=6, N=20)

            feature_label = ["feature"]
            dim_label = ["first"]
            cnst_train = Constant(
                rg=Generator(PCG64(77)),
                dim_label=dim_label,
                feature_label=feature_label,
            )

            rolling = [
                RollingWindow(
                    axis="ID",
                    train_size=2,
                    val_size=1,
                    test_size=1,
                    step_size=1,
                )
            ]

            cfg = _make_config(
                self.DATA_PATH,
                models_generate=[],
                models_train=[cnst_train],
                models_forecast=[cnst_train],
                rolling=rolling,
            )
            exp = Experiment(cfg)
            exp.run(train=True, forecast=True)
        finally:
            _cleanup(self.DATA_PATH)

    def test_rolling_window_timestamp_axis(self):
        """Timestamp-axis walk-forward: forecasts land on held-out timestamps,
        so metrics pair and ``results`` populate."""
        try:
            cnst = Constant(
                rg=Generator(PCG64(7)),
                dim_label=["first"],
                feature_label=["feature"],
            )
            rolling = [
                RollingWindow(
                    axis="timestamp",
                    train_size=30,
                    val_size=10,
                    test_size=10,
                    step_size=10,
                )
            ]
            cfg = ExperimentConfig(
                general=General(data_path=self.DATA_PATH, n_jobs=1),
                generate=Stage(models=[cnst], params={"N": 60}),
                train=Stage(models=[cnst], rolling=rolling),
                forecast=Stage(models=[cnst]),
                output=Output(metrics=[mae, rmse]),
            )
            exp = Experiment(cfg)
            exp.run()  # auto-detect: initialize + generate + train + forecast + output

            # Walk-forward forecasts overlap the actuals -> non-empty results.
            assert exp.results
            metric_dicts = [
                scores
                for models in exp.results.values()
                for scores in models.values()
            ]
            assert metric_dicts
            assert all(
                "mae" in scores and "rmse" in scores for scores in metric_dicts
            )
        finally:
            _cleanup(self.DATA_PATH)

    def test_rolling_window_timestamp_expanding(self):
        """Expanding timestamp window runs and produces paired results."""
        try:
            cnst = Constant(
                rg=Generator(PCG64(3)),
                dim_label=["first"],
                feature_label=["feature"],
            )
            rolling = [
                RollingWindow(
                    axis="timestamp",
                    train_size=10,
                    val_size=0,
                    test_size=5,
                    step_size=5,
                    expanding=True,
                )
            ]
            cfg = ExperimentConfig(
                general=General(data_path=self.DATA_PATH, n_jobs=1),
                generate=Stage(models=[cnst], params={"N": 60}),
                train=Stage(models=[cnst], rolling=rolling),
                forecast=Stage(models=[cnst]),
                output=Output(metrics=[mae, rmse]),
            )
            exp = Experiment(cfg)
            exp.run()
            assert exp.results
        finally:
            _cleanup(self.DATA_PATH)

    def test_rolling_window_nested_id_timestamp(self):
        """Nested roll: train per day (ID), forecast within each test day
        (timestamp axis, frozen model via retrain=False) -> results populate."""
        try:
            self._setup_multi_id_data(self.DATA_PATH, n_ids=4, N=12)
            cnst = Constant(
                rg=Generator(PCG64(7)),
                dim_label=["first"],
                feature_label=["feature"],
            )
            rolling = [
                RollingWindow(
                    axis="ID", train_size=2, val_size=0, test_size=1,
                    step_size=1, retrain=True,
                ),
                RollingWindow(
                    axis="timestamp", train_size=4, val_size=0, test_size=1,
                    step_size=1, retrain=False, expanding=True,
                ),
            ]
            cfg = ExperimentConfig(
                general=General(data_path=self.DATA_PATH, n_jobs=1),
                generate=Stage(models=[]),
                train=Stage(models=[cnst], rolling=rolling),
                forecast=Stage(models=[cnst]),
                output=Output(metrics=[mae, rmse]),
            )
            exp = Experiment(cfg)
            # Data is pre-set up; run train+forecast+output (no initialize/generate).
            exp.run(train=True, forecast=True, output=True)

            assert exp.results
            metric_dicts = [
                scores
                for models in exp.results.values()
                for scores in models.values()
            ]
            assert metric_dicts
            assert all(
                "mae" in scores and "rmse" in scores for scores in metric_dicts
            )
        finally:
            _cleanup(self.DATA_PATH)


# ---------------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------------

class TestExperimentDevice:
    """Test device configuration."""

    DATA_PATH = "data/test_experiment_device/"

    def test_cpu_device(self):
        """Setting device='cpu' sets CUDA_VISIBLE_DEVICES."""
        import os

        feature_label = ["feature"]
        dim_label = ["first"]
        cnst = Constant(
            rg=Generator(PCG64(1234)),
            dim_label=dim_label,
            feature_label=feature_label,
        )
        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[cnst],
            models_train=[cnst],
            models_forecast=[cnst],
            device="cpu",
        )
        try:
            Experiment(cfg)
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
        finally:
            _cleanup(self.DATA_PATH)

    def test_gpu_device(self):
        """Setting device='gpu:1' sets CUDA_VISIBLE_DEVICES to '1'."""
        import os

        feature_label = ["feature"]
        dim_label = ["first"]
        cnst = Constant(
            rg=Generator(PCG64(1234)),
            dim_label=dim_label,
            feature_label=feature_label,
        )
        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[cnst],
            models_train=[cnst],
            models_forecast=[cnst],
            device="gpu:1",
        )
        try:
            Experiment(cfg)
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
        finally:
            _cleanup(self.DATA_PATH)

    def test_gpu_default_device(self):
        """Setting device='gpu' defaults to GPU 0."""
        import os

        feature_label = ["feature"]
        dim_label = ["first"]
        cnst = Constant(
            rg=Generator(PCG64(1234)),
            dim_label=dim_label,
            feature_label=feature_label,
        )
        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[cnst],
            models_train=[cnst],
            models_forecast=[cnst],
            device="gpu",
        )
        try:
            Experiment(cfg)
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        finally:
            _cleanup(self.DATA_PATH)


# ---------------------------------------------------------------------------
# ID-wise generation
# ---------------------------------------------------------------------------

class TestExperimentIDWise:
    """Test ID-wise generation mode."""

    DATA_PATH = "data/test_experiment_idwise/"

    def test_id_wise_generate(self):
        """ID-wise generation completes."""
        feature_label = ["feature"]
        dim_label = ["first"]
        cnst = Constant(
            rg=Generator(PCG64(1234)),
            dim_label=dim_label,
            feature_label=feature_label,
        )
        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[cnst],
            models_train=[cnst],
            models_forecast=[cnst],
            id_wise=True,
        )
        try:
            exp = Experiment(cfg)
            exp.run(initialize=True, generate=True, train=True, forecast=True)
        finally:
            _cleanup(self.DATA_PATH)


# ---------------------------------------------------------------------------
# compute_metrics direct test
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Test compute_metrics method directly."""

    DATA_PATH = "data/test_experiment_compute/"

    def test_compute_metrics_directly(self):
        """compute_metrics returns correct values."""

        def mae(y_true, y_pred):
            return np.mean(np.abs(y_true - y_pred))

        feature_label = ["feature"]
        dim_label = ["first"]
        cnst = Constant(
            rg=Generator(PCG64(1234)),
            dim_label=dim_label,
            feature_label=feature_label,
        )
        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[cnst],
            models_train=[cnst],
            models_forecast=[cnst],
            metrics=[mae],
        )
        try:
            exp = Experiment(cfg)
            y_true = np.array([1.0, 2.0, 3.0])
            y_pred = np.array([1.5, 2.5, 3.5])
            result = exp.compute_metrics(y_true, y_pred)
            assert "mae" in result
            assert abs(result["mae"] - 0.5) < 1e-10
        finally:
            _cleanup(self.DATA_PATH)

    def test_compute_metrics_empty(self):
        """compute_metrics returns empty dict when no metrics configured."""
        feature_label = ["feature"]
        dim_label = ["first"]
        cnst = Constant(
            rg=Generator(PCG64(1234)),
            dim_label=dim_label,
            feature_label=feature_label,
        )
        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[cnst],
            models_train=[cnst],
            models_forecast=[cnst],
        )
        try:
            exp = Experiment(cfg)
            result = exp.compute_metrics(np.array([1.0]), np.array([1.0]))
            assert result == {}
        finally:
            _cleanup(self.DATA_PATH)

    def test_results_property_before_run(self):
        """results property returns empty dict before run."""
        feature_label = ["feature"]
        dim_label = ["first"]
        cnst = Constant(
            rg=Generator(PCG64(1234)),
            dim_label=dim_label,
            feature_label=feature_label,
        )
        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[cnst],
            models_train=[cnst],
            models_forecast=[cnst],
        )
        try:
            exp = Experiment(cfg)
            assert exp.results == {}
        finally:
            _cleanup(self.DATA_PATH)

    def test_get_output_loader(self):
        """get_output_loader returns a loader."""
        feature_label = ["feature"]
        dim_label = ["first"]
        cnst = Constant(
            rg=Generator(PCG64(1234)),
            dim_label=dim_label,
            feature_label=feature_label,
        )
        cfg = _make_config(
            self.DATA_PATH,
            models_generate=[cnst],
            models_train=[cnst],
            models_forecast=[cnst],
        )
        try:
            exp = Experiment(cfg)
            loader = exp.get_output_loader()
            assert loader is not None
        finally:
            _cleanup(self.DATA_PATH)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
