import pytest
import numpy as np
from TSbench import TSmodels, LoaderTSdf
from numpy.random import Generator, PCG64

def is_reproduced(loader):
    path = "data/test_models/"
    datatype = "simulated"
    reproduce_loader = LoaderTSdf(path=path, datatype=datatype)
    assert loader.get_df().equals(reproduce_loader.get_df())

def test_simple():
    seed = 1234
    N = 10
    T = 5
    feature_label = ["feature"]
    rg = Generator(PCG64(seed))
    # loader
    path = "data/"
    datatype = "simulated"
    loader = LoaderTSdf(path=path, datatype=datatype)
    # Simple model
    cnst_model = TSmodels.Constant(dim=1, rg=rg,
                                feature_label=feature_label)
    # generate
    cnst_model.generate(N)
    ID = str(cnst_model)
    cnst_model.register_data(loader)
    ## forecast IS
    timeseries = loader.get_timeseries(IDs=[ID], end=3, features=["feature"])
    cnst_model.set_data(timeseries)
    cnst_model.forecast(T)
    cnst_model.register_data(loader, append_to_feature=str(cnst_model))
    ## forecast OOS
    timeseries = loader.get_timeseries(IDs=[ID], end_index=10, features=["feature"])
    cnst_model.set_data(timeseries)
    cnst_model.forecast(T)
    cnst_model.register_data(loader, append_to_feature=str(cnst_model))
    ## rolling forecast OOS
    timeseries = loader.get_timeseries(IDs=[ID], end_index=10, features=["feature"])
    cnst_model.set_data()
    x = cnst_model.rolling_forecast(T)

    #is_reproduced(loader)
test_simple()

def test_set_data():
    seed = 1234
    N = 10
    T = 5
    feature_label = ["feature"]
    rg = Generator(PCG64(seed))
    # loader
    path = "data/"
    datatype = "simulated"
    loader = LoaderTSdf(path=path, datatype=datatype)
    # Simple model
    cnst_model = TSmodels.Constant(dim=1, rg=rg,
                                feature_label=feature_label)
    # generate 1
    x0 = [np.array([0, 1 , 2])]
    cnst_model.set_data(data=x0, reset_timestamp = True)
    g1 = cnst_model.generate(N)
    # generate 2
    x0 = [0, 1 , 2]
    cnst_model.set_data(data=x0, reset_timestamp = True)
    g2 = cnst_model.generate(N)
    ## generate 3
    g3 = cnst_model.generate(N, reset_timestamp=True)
    ## generate 4
    cnst_model.set_data()
    g4 = cnst_model.generate(N)
    #assert all(g1 == g2)
    #assert all(g3 == g4)

def test_rolling_forecast():
    seed = 1234
    N = 10
    T = 5
    feature_label = ["feature"]
    rg = Generator(PCG64(seed))
    # loader
    path = "data/"
    datatype = "simulated"
    loader = LoaderTSdf(path=path, datatype=datatype)
    # Simple model
    cnst_model = TSmodels.Constant(dim=1, rg=rg,
                                feature_label=feature_label)
    # generate
    cnst_model.generate(N)
    ID = str(cnst_model)
    cnst_model.register_data(loader)
    ## forecast IS
    timeseries = loader.get_timeseries(IDs=[ID], end=3, features=["feature"])
    cnst_model.set_data(timeseries)
    cnst_model.forecast(T)
    cnst_model.register_data(loader, append_to_feature=str(cnst_model))
    ## forecast OOS
    timeseries = loader.get_timeseries(IDs=[ID], end_index=10, features=["feature"])
    cnst_model.set_data(timeseries)
    cnst_model.forecast(T)
    cnst_model.register_data(loader, append_to_feature=str(cnst_model))

    is_reproduced(loader)

def test_parametersIO():
    seed = 1234
    N = 10
    T = 5
    feature_label = ["feature"]
    rg = Generator(PCG64(seed))
    # loader
    path = "data/"
    datatype = "simulated"
    loader = LoaderTSdf(path=path, datatype=datatype)
    # Simple model
    cnst_model = TSmodels.Constant(lag=1, dim=1, rg=rg,
                                feature_label=feature_label)
    # save model
    cnst_model.save_model("cnst.pkl")
    cnst_model.save_model("cnst.json")
    with pytest.raises(ValueError):
        cnst_model.save_model("cnst.misc")

    # load model
    pkl_model = cnst_model.load_model("cnst.pkl")
    json_model = TSmodels.Constant.load_model("cnst.json")
    assert all(cnst_model.constant == pkl_model.constant)
    assert all(cnst_model.constant == json_model.constant)

@pytest.mark.R
def test_R():
    """Using R packages"""
    TSmodels.rGARCH()

def test_arma():
    """Comparisson with statmodels."""
    ar = [0.5, 0.33]
    ma = [0.5, 0.6]
    seed = 1234
    N = 10
    rg = Generator(PCG64(seed))
    arma_model = TSmodels.ARMA(ar=ar, ma=ma, rg=rg)
    x0 = [0, 1, 2]
    arma_model.set_data(data=x0)
    arma_model.generate(N=N)

def test_set_data():
    """TSmodel.Constant uses set_data in generate."""
    ar = [0.5, 0.33]
    ma = [0.5, 0.6]
    seed = 1234
    T = 10
    rg = Generator(PCG64(seed))
    cnst_model = TSmodels.Constant()
    x0 = [np.array([0, 1 , 2])]
    cnst_model.generate(N=N, data=x0)
    x0 = np.array([0, 1 , 2])
    cnst_model.generate(N=N, data=x0)
    x0 = [0, 1, 2]
    cnst_model.generate(N=N, data=x0)
    x0 = []
    cnst_model.generate(N=N, data=x0)
    x0 = None
    cnst_model.generate(N=N, data=x0)

test_arma()
