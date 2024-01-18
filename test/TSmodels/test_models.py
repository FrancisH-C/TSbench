import pytest
import numpy as np
from TSbench import TSmodels, LoaderTSdf
from numpy.random import Generator, PCG64

def is_reproduced(loader):
    path = "data/test_models/"
    datatype = "simulated"
    reproduce_loader = LoaderTSdf(path=path, datatype=datatype)
    assert loader.get_df().equals(reproduce_loader.get_df())

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
    x0 = np.array([0, 1 , 2])
    cnst_model.set_data(data=x0)
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
    ## generate 5
    cnst_model.set_data([])
    g5 = cnst_model.generate(N)
    ## generate 6
    cnst_model.set_data(None)
    g6 = cnst_model.generate(N)

    # with initial data
    assert all(g1 == g2)
    # no initial data
    assert all(g3 == g4)
    assert all(g4 == g5)
    assert all(g5 == g6)

def test_forecast():
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
    # Generate
    cnst_model.generate(N)
    ID = str(cnst_model)
    cnst_model.register_data(loader)

    # Test Forecast
    # forecast IS
    timeseries = loader.get_timeseries(IDs=[ID], end=3, features=["feature"])
    cnst_model.set_data(timeseries)
    cnst_model.forecast(T, reset_timestamp=False, collision="overwrite")
    cnst_model.register_data(loader, append_to_feature=str(cnst_model))
    ## forecast OOS
    timeseries = loader.get_timeseries(IDs=[ID], end_index=10, features=["feature"])
    cnst_model.set_data(timeseries)
    cnst_model.forecast(T, reset_timestamp=False, collision="overwrite")
    cnst_model.register_data(loader, append_to_feature=str(cnst_model))
    # Test
    is_reproduced(loader)

    # Test rolling forecast
    loader.rm_feature("feature_Constant")
    ## rolling forecast IS
    timeseries = loader.get_timeseries(IDs=[ID], end=3, features=["feature"])
    cnst_model.set_data(timeseries)
    cnst_model.rolling_forecast(T)
    cnst_model.register_data(loader, append_to_feature=str(cnst_model))
    ## rolling forecast OOS
    timeseries = loader.get_timeseries(IDs=[ID], end_index=10, features=["feature"])
    cnst_model.set_data(timeseries)
    cnst_model.rolling_forecast(T)
    cnst_model.register_data(loader, append_to_feature=str(cnst_model))
    # Test
    is_reproduced(loader)

    # Rolling forecast parameters
    ## batch_size > window_size
    T = 14
    batch_size = 5
    window_size = 2
    timeseries = loader.get_timeseries(IDs=["Constant"], start_index=0, end_index=10, features=["feature"])
    cnst_model.set_data(timeseries)
    cnst_model.rolling_forecast(T, batch_size=batch_size, window_size=window_size)
    assert cnst_model.get_data().size == 10
    ### train == True
    cnst_model.set_data(timeseries)
    cnst_model.rolling_forecast(T, train=True, batch_size=batch_size, window_size=window_size)
    ## batch_size > T
    T = 7
    batch_size = 9
    window_size = 2
    timeseries = loader.get_timeseries(IDs=["Constant"], start_index=0, end_index=10, features=["feature"])
    cnst_model.set_data(timeseries)
    cnst_model.rolling_forecast(T, batch_size=batch_size, window_size=window_size)
    assert cnst_model.get_data().size == 0
    #### side == after
    cnst_model.set_data(timeseries)
    cnst_model.rolling_forecast(T, batch_size=batch_size, window_size=window_size, side="after")
    assert cnst_model.get_data().size == 9
    cnst_model.register_data(loader, append_to_feature="Constant_rolling")

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
    cnst_model.save_model("data/cnst.pkl")
    cnst_model.save_model("data/cnst.json")
    with pytest.raises(ValueError):
        cnst_model.save_model("data/cnst.misc")

    # load model
    pkl_model = cnst_model.load_model("data/cnst.pkl")
    json_model = TSmodels.Constant.load_model("data/cnst.json")
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
    T = 10
    rg = Generator(PCG64(seed))
    arma_model = TSmodels.ARMA(ar=ar, ma=ma, d=1, rg=rg)
    x0 = [0, 1, 2]
    #arma_model.set_data(data=x0)
    timeseries = arma_model.generate(N=N)
    arma_model.set_data(data=timeseries)
    arma_model.train()
    timeseries = arma_model.forecast(T=T)
    # dim=2
    #arma_model = TSmodels.ARMA(dim=2, lag=2, rg=rg)
    #timeseries = arma_model.generate(N=N)
    #arma_model.set_data(data=timeseries)
    #arma_model.train()
    #arma_model.forecast(T=T)

def test_GARCH():
    """Comparisson with statmodels."""
    lag = 3
    N = 10
    rg = Generator(PCG64(1234))
    garch_model = TSmodels.GARCH(lag=lag, rg=rg)
    timeseries = garch_model.generate(N=N)

#def test_VEC_GARCH():
#    """Comparisson with statmodels."""
#    seed = 9103
#    dim = 2
#    lag = 3
#    N = 10
#    rg = Generator(PCG64(seed))
#    garch_model = TSmodels.VEC_GARCH(dim=2, lag=lag, rg=rg)
#    #data = [np.array([1.0,1,1]), np.array([1.0,2,3])]
#    #garch_model.set_data(data=data)
#    timeseries = garch_model.generate(N=N)
#    print(timeseries)

test_arma()
