"""Tests for the typed dataclass Experiment config (``ExperimentConfig``)."""

import shutil

import numpy as np
import pytest
from numpy.random import Generator, PCG64

from TSbench.experiment import Experiment, ExperimentConfig, General, Stage, Output
from TSbench.TSdata.TSloader import LoaderTSdf
from TSbench.TSmodels import Constant


def _cleanup(path):
    shutil.rmtree(path, ignore_errors=True)


def _constant(seed=1234, name=None):
    return Constant(
        rg=Generator(PCG64(seed)),
        dim_label=["first"],
        feature_label=["feature"],
        name=name,
    )


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_output_path_defaults_to_data_path(self):
        g = General(data_path="data/foo/")
        assert g.output_path == "data/foo/"

    def test_general_defaults(self):
        g = General()
        assert g.datatype == "simulated"
        assert g.n_jobs == 1
        assert g.n_input_loaders == 1
        assert g.device is None

    def test_stage_defaults(self):
        s = Stage()
        assert s.models == []
        assert s.params == {}
        assert s.input_loaders_params == {}
        assert s.id_wise is False
        assert s.rolling_window is None

    def test_experiment_config_defaults(self):
        cfg = ExperimentConfig()
        assert isinstance(cfg.general, General)
        assert isinstance(cfg.generate, Stage)
        assert isinstance(cfg.output, Output)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_data_path_raises(self):
        with pytest.raises(ValueError):
            General(data_path="")

    def test_bad_n_jobs_raises(self):
        with pytest.raises(ValueError):
            General(n_jobs=0)

    def test_metrics_not_a_list_raises(self):
        with pytest.raises(ValueError):
            Output(metrics="not a list")  # type: ignore[arg-type]

    def test_non_callable_metric_raises(self):
        with pytest.raises(ValueError):
            Output(metrics=[42])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Dataclass-native wiring through Experiment
# ---------------------------------------------------------------------------

class TestWiring:
    """``Experiment`` consumes the config dataclasses directly as its stages."""

    DATA_PATH = "data/test_config_wiring/"

    def _experiment(self, **kwargs):
        kwargs.setdefault("general", General(data_path=self.DATA_PATH))
        return Experiment(ExperimentConfig(**kwargs))

    def test_id_wise_wired(self):
        try:
            exp = self._experiment(generate=Stage(models=[], id_wise=True))
            assert exp.generate.id_wise is True
        finally:
            _cleanup(self.DATA_PATH)

    def test_rolling_window_wired(self):
        rw = {"train_size": 1, "val_size": 1, "test_size": 1}
        try:
            exp = self._experiment(train=Stage(rolling_window=rw))
            assert exp.train.rolling_window == rw
        finally:
            _cleanup(self.DATA_PATH)

    def test_metrics_wired(self):
        try:
            exp = self._experiment(output=Output(metrics=[_mae]))
            assert exp.output.metrics == [_mae]
        finally:
            _cleanup(self.DATA_PATH)

    def test_default_initialize_set(self):
        try:
            exp = self._experiment()
            assert callable(exp.initialize)
        finally:
            _cleanup(self.DATA_PATH)

    def test_custom_initialize_preserved(self):
        def sentinel():
            return None

        try:
            exp = self._experiment(initialize=sentinel)
            assert exp.initialize is sentinel
        finally:
            _cleanup(self.DATA_PATH)

    def test_device_wired(self):
        try:
            exp = self._experiment(
                general=General(data_path=self.DATA_PATH, device="cpu")
            )
            assert exp.general.device == "cpu"
        finally:
            _cleanup(self.DATA_PATH)

    def test_non_config_raises(self):
        with pytest.raises(TypeError):
            Experiment({"general": {}})


# ---------------------------------------------------------------------------
# End-to-end through Experiment
# ---------------------------------------------------------------------------

class TestEndToEnd:
    DATA_PATH = "data/test_config_e2e/"

    def test_auto_detect_run(self):
        """``run()`` with no flags runs every configured stage."""
        cnst = _constant()
        cfg = ExperimentConfig(
            general=General(data_path=self.DATA_PATH),
            generate=Stage(models=[cnst], params={"N": 20}),
            train=Stage(models=[cnst]),
            forecast=Stage(models=[cnst], params={"T": 5}),
            output=Output(metrics=[_mae]),
        )
        try:
            exp = Experiment(cfg)
            exp.run()  # auto-detect: init + generate + train + forecast + output
            loader = LoaderTSdf(datatype="simulated", path=self.DATA_PATH)
            assert len(loader.get_df()) > 0
            assert isinstance(exp.results, dict)
        finally:
            _cleanup(self.DATA_PATH)

    def test_explicit_partial_run(self):
        """Explicit flags select a subset; others default to False."""
        cnst = _constant()
        cfg = ExperimentConfig(
            general=General(data_path=self.DATA_PATH),
            generate=Stage(models=[cnst], params={"N": 20}),
            forecast=Stage(models=[cnst], params={"T": 5}),
        )
        try:
            exp = Experiment(cfg)
            exp.run(initialize=True, generate=True)
            df = LoaderTSdf(datatype="simulated", path=self.DATA_PATH).get_df()
            assert len(df) > 0
            # generate-only: no forecast columns (no "base_model" features)
            assert not any("_" in col for col in df.columns)
        finally:
            _cleanup(self.DATA_PATH)

    def test_empty_config_runs(self):
        """A zero-model config still constructs and runs without error."""
        try:
            exp = Experiment(
                ExperimentConfig(general=General(data_path=self.DATA_PATH))
            )
            exp.run()
            assert exp.results == {}
        finally:
            _cleanup(self.DATA_PATH)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
