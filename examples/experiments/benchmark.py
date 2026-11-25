"""Generate-and-benchmark: score forecasters against a known DGP.

An ARMA process is the *data-generating process* (the "truth"). We hold out
the tail of the series, train a Constant baseline and an ARMA model on the
rest, forecast the held-out tail with each, and score the forecasts against
the actual values with MAE / RMSE.

Run from the repo root::

    python examples/experiments/benchmark.py

The trick that makes metrics meaningful: feed each forecaster only the first
``TRAIN_END`` points (``input_loaders_params={"end_index": TRAIN_END}``) so its
forecast lands on the held-out tail, overlapping the actuals and getting scored.
"""

from pathlib import Path

from numpy.random import Generator, PCG64

from TSbench.experiment import Experiment, ExperimentConfig, General, Stage, Output
from TSbench.metrics import mae, rmse
from TSbench.TSmodels import ARMA, Constant

DATA_PATH = str(Path(__file__).parent / "data" / "benchmark")

N = 100  # length of the generated series
HOLDOUT = 20  # last HOLDOUT points are forecast and scored
TRAIN_END = N - HOLDOUT

# Data-generating process (the "truth").
dgp = ARMA(
    lag=1,
    rg=Generator(PCG64(1)),
    dim_label=["d0"],
    feature_label=["feature"],
    name="series",
)

# Competing forecasters.
constant = Constant(rg=Generator(PCG64(2)), dim_label=["d0"], feature_label=["feature"])
arma = ARMA(lag=1, rg=Generator(PCG64(3)), dim_label=["d0"], feature_label=["feature"])

config = ExperimentConfig(
    general=General(data_path=DATA_PATH),
    generate=Stage(models=[dgp], params={"N": N}),
    train=Stage(models=[constant, arma], input_loaders_params={"end_index": TRAIN_END}),
    forecast=Stage(
        models=[constant, arma],
        params={"T": HOLDOUT},
        input_loaders_params={"end_index": TRAIN_END},
    ),
    output=Output(metrics=[mae, rmse]),
)


if __name__ == "__main__":
    exp = Experiment(config)
    exp.run()
    # exp.results: {ID: {model: {metric_name: value}}}
    print(exp.results)
    for ID, per_model in exp.results.items():
        print(f"series {ID!r}:")
        for model, scores in per_model.items():
            line = "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
            print(f"  {model:10s} {line}")
