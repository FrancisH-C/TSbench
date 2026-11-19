"""Minimal TSbench experiment: generate a series, then forecast it.

A TSbench config is just Python — there's no special location or registry.
Put a file like this anywhere and run it::

    python examples/experiments/minimal.py

The only path TSbench reads/writes is the dataset directory named in
``General(data_path=...)`` (here, anchored next to this file). Each run
restarts that dataset from scratch.
"""

from pathlib import Path

from numpy.random import Generator, PCG64

from TSbench.experiment import Experiment, ExperimentConfig, General, Stage
from TSbench.TSdata.TSloader import LoaderTSdf
from TSbench.TSmodels import ARMA

DATA_PATH = str(Path(__file__).parent / "data" / "minimal")

arma = ARMA(
    lag=1, rg=Generator(PCG64(1234)), dim_label=["d0"], feature_label=["feature"]
)

config = ExperimentConfig(
    general=General(data_path=DATA_PATH),
    generate=Stage(models=[arma], params={"N": 100}),
    train=Stage(models=[arma]),
    forecast=Stage(models=[arma], params={"T": 10}),
)


if __name__ == "__main__":
    exp = Experiment(config)
    exp.run()  # no flags = initialize -> generate -> train -> forecast -> output
    df = LoaderTSdf(datatype="simulated", path=DATA_PATH).get_df()
    print(df.tail(15))
    # For scoring forecasts against actuals, see benchmark.py.
