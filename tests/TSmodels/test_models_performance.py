import timeit

import pytest
from numpy.random import PCG64, Generator

from TSbench import TSmodels
from TSbench.TSdata import LoaderTSdf


@pytest.mark.performance
def test_arma():
    seed = 1234
    N = 60
    T = 10
    d = 1
    for dim in range(1, 5):
        for lag in range(1, 5):
            rg = Generator(PCG64(seed))
            arma_model = TSmodels.ARMA(dim=dim, lag=lag, d=d, rg=rg)
            timeseries = arma_model.generate(N=N)
            arma_model.set_data(data=timeseries)
            timeseries = arma_model.generate(N=N)
            arma_model.train()
            timeseries = arma_model.forecast(T=T)


@pytest.mark.performance
def test_forecast_performance():
    def make_forecast(T_forecast):
        ID = "Constant"
        path = "data/"
        datatype = "simulated"
        feature_label = ["feature"]
        loader = LoaderTSdf(path=path, datatype=datatype)
        cnst_model = TSmodels.Constant(feature_label=feature_label)

        T = 10
        cnst_model.generate(T)
        cnst_model.register_data(loader)
        df = loader.get_df(start=0, end=5)
        cnst_model.set_data(data=df)
        cnst_model.train()

        df = loader.get_df(IDs=[ID], end=5, features=feature_label)
        cnst_model.set_data(data=df)
        cnst_model.forecast(T=T_forecast)

    print(timeit.timeit(lambda: make_forecast(10000), number=1))
    print(timeit.timeit(lambda: make_forecast(100000), number=1))
    print(timeit.timeit(lambda: make_forecast(1000000), number=1))
    print(timeit.timeit(lambda: make_forecast(10000000), number=1))
    # cProfile.run('timeit.timeit(lambda: foo(10000000), number=1)')
