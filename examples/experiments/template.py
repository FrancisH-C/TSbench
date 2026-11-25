"""Annotated template: every knob of the dataclass experiment config.

This is a copy-paste starting point. It runs the full generate-and-benchmark
pipeline (generate a DGP, train + forecast two competitors on a held-out tail,
score with metrics) and also demonstrates the *advanced escape hatches* that
``minimal.py`` and ``benchmark.py`` don't:

- a custom ``initialize`` callable (prepare the dataset before the run),
- an ``Output`` transform (``process_df``) applied to the final results,
- inline notes for the remaining knobs (``device``, ``id_wise``,
  ``rolling``, ``input_loaders_params``, and the ``Process``-based
  ``pre_process``/``run_process`` stages).

Run from the repo root::

    python examples/experiments/template.py

A config is just Python — copy this file anywhere, edit it, and run it.
"""

from pathlib import Path

from numpy.random import Generator, PCG64

from TSbench.experiment import Experiment, ExperimentConfig, General, Stage, Output
from TSbench.metrics import mae, rmse
from TSbench.TSdata.TSloader import LoaderTSdf
from TSbench.TSmodels import ARMA, Constant

DATA_PATH = str(Path(__file__).parent / "data" / "template")

N = 100              # length of the generated series
HOLDOUT = 20         # last HOLDOUT points are forecast and scored
TRAIN_END = N - HOLDOUT

# Models double as data-generating process and forecasters.
dgp = ARMA(
    lag=1, rg=Generator(PCG64(1)), dim_label=["d0"],
    feature_label=["feature"], name="series",
)
constant = Constant(
    rg=Generator(PCG64(2)), dim_label=["d0"], feature_label=["feature"]
)
arma = ARMA(
    lag=1, rg=Generator(PCG64(3)), dim_label=["d0"], feature_label=["feature"]
)


# --- advanced escape hatch 1: a custom initialize callable ------------------
# Runs before the pipeline to prepare the dataset. Omit `initialize=` on the
# config to get the default (restart an empty dataset at `data_path`).
def initialize():
    LoaderTSdf(datatype="simulated", path=DATA_PATH).restart_dataset()


# --- advanced escape hatch 2: an output transform (process_df) --------------
# Applied per ID during the output stage (after forecasting, when data exists).
# Signature is `(df) -> df` and it must return a DataFrame.
def round_results(df):
    return df.round(6)


config = ExperimentConfig(
    general=General(
        data_path=DATA_PATH,
        # output_path=DATA_PATH,   # defaults to data_path
        datatype="simulated",
        n_jobs=1,                  # parallel jobs per loader
        n_input_loaders=1,         # number of input loaders
        # device="cpu",            # "cpu" | "gpu" | "gpu:N" -> CUDA_VISIBLE_DEVICES
    ),
    initialize=initialize,         # optional; omit for the default restart
    generate=Stage(
        models=[dgp],
        params={"N": N},
        # id_wise=False,           # generate once per ID instead of once total
        # input_loaders_params={}, # extra kwargs forwarded to input_loader.get_df
    ),
    train=Stage(
        models=[constant, arma],
        input_loaders_params={"end_index": TRAIN_END},  # train on the head only
        # Rolling-window training (from TSbench.experiment import RollingWindow).
        # A list: its order is the nesting order, outermost first.
        # rolling=[
        #     RollingWindow(axis="ID", train_size=1, val_size=1, test_size=1),
        # ],
    ),
    forecast=Stage(
        models=[constant, arma],
        params={"T": HOLDOUT},
        input_loaders_params={"end_index": TRAIN_END},  # forecast the held-out tail
    ),
    output=Output(
        metrics=[mae, rmse],
        process_df=round_results,  # advanced: transform the final results per ID
        # process_split=...,       # advanced: per-split hook, (input_loader) -> None
    ),
    # --- advanced escape hatch 3: pre/run-process stages --------------------
    # Provide a Process to customize the raw-input or run loaders. Each takes
    # process_df=(df)->df, process_split=(input_loader)->None, output_loader=...
    # from TSbench.experiment import Process
    # pre_process=Process(process_split=my_hook),
    # run_process=Process(output_loader=my_loader),
)


if __name__ == "__main__":
    exp = Experiment(config)
    exp.run()  # no flags = run every configured stage
    # exp.results: {ID: {model: {metric_name: value}}}
    for ID, per_model in exp.results.items():
        print(f"series {ID!r}:")
        for model, scores in per_model.items():
            line = "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
            print(f"  {model:10s} {line}")
