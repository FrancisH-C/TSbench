import numpy as np
import pytest
from numpy.random import PCG64, Generator

from TSbench import TSmodels
from TSbench.TSdata import LoaderTSdf


def test_data_to_reprodue():
    path = "data/test_models"
    datatype = "simulated"
    loader = LoaderTSdf(path=path, datatype=datatype)

    seed = 1234
    ID = ["Constant"]
    feature_label = ["feature"]
    cnst_model = TSmodels.Constant(
        dim=1, rg=Generator(PCG64(seed)), feature_label=feature_label
    )

    T = 10
    cnst_model.generate(T)
    cnst_model.register_data(loader, collision="overwrite")

    timeseries = loader.get_timeseries(IDs=ID, end_index=4, features=feature_label)
    cnst_model.set_data(data=timeseries)

    timeseries = loader.get_timeseries(IDs=ID, end=[3], features=feature_label)
    cnst_model.set_data(data=timeseries)
    T = 5
    cnst_model.forecast(T)
    cnst_model.register_data(loader, append_to_feature=str(cnst_model))

    timeseries = loader.get_timeseries(IDs=ID, end_index=10, features=feature_label)
    cnst_model.set_data(data=timeseries)
    T = 5
    cnst_model.forecast(T)
    cnst_model.register_data(loader, append_to_feature=str(cnst_model))

    loader.write()


def is_reproduced(loader):
    path = "data/test_models/"
    datatype = "simulated"
    reproduce_loader = LoaderTSdf(path=path, datatype=datatype)
    assert loader.get_df().equals(reproduce_loader.get_df())


def test_model_generate():
    for dim in range(1, 10):
        seed = 1234
        N = 10
        feature_label = ["feature"]
        rg = Generator(PCG64(seed))
        # loader
        # Simple model
        cnst_model = TSmodels.Constant(dim=dim, rg=rg, feature_label=feature_label)
        # generate 1
        if dim == 1:
            data = np.array([0, 1, 2])
        if dim >= 2:
            data = np.ones((2, dim))
        cnst_model.set_data(data=data)
        cnst_model.set_data(data=data, reset_timestamp=True)
        g1 = cnst_model.generate(N)
        # generate 2
        if dim == 1:
            data = [0, 1, 2]
        if dim >= 2:
            data = [np.ones((2, dim))]
        cnst_model.set_data(data=data, reset_timestamp=True)
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


def test_model_forecast_indexing():
    seed = 1234
    N = 10
    T = 5
    dim = 1
    feature_label = ["feature"]
    rg = Generator(PCG64(seed))
    # loader
    path = "data/"
    datatype = "simulated"
    loader = LoaderTSdf(path=path, datatype=datatype)
    # Simple model
    cnst_model = TSmodels.Constant(dim=dim, rg=rg, feature_label=feature_label)
    # Generate
    cnst_model.generate(N)
    ID = str(cnst_model)
    cnst_model.register_data(loader)

    # Test Forecast
    # forecast IS
    timeseries = loader.get_timeseries(IDs=[ID], end=[3], features=["feature"])
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
    timeseries = loader.get_timeseries(IDs=[ID], end=[3], features=["feature"])
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
    timeseries = loader.get_timeseries(
        IDs=["Constant"], start_index=0, end_index=10, features=["feature"]
    )
    cnst_model.set_data(timeseries)
    cnst_model.rolling_forecast(T, batch_size=batch_size, window_size=window_size)
    assert cnst_model.get_data().size == 10
    ### train == True
    cnst_model.set_data(timeseries)
    cnst_model.rolling_forecast(
        T, train=True, batch_size=batch_size, window_size=window_size
    )
    ## batch_size > T
    T = 7
    batch_size = 9
    window_size = 2
    timeseries = loader.get_timeseries(
        IDs=["Constant"], start_index=0, end_index=10, features=["feature"]
    )
    cnst_model.set_data(timeseries)
    cnst_model.rolling_forecast(T, batch_size=batch_size, window_size=window_size)
    assert cnst_model.get_data().size == 0
    #### side == after
    cnst_model.set_data(timeseries)
    cnst_model.rolling_forecast(
        T, batch_size=batch_size, window_size=window_size, side="after"
    )
    assert cnst_model.get_data().size == 9
    cnst_model.register_data(loader, append_to_feature="Constant_rolling")


def test_model_forecast():
    seed = 1234
    N = 10
    T = 5
    dim = 1
    for dim in range(1, 6):
        feature_label = ["feature"]
        rg = Generator(PCG64(seed))
        # loader
        path = "data/"
        datatype = "simulated"
        loader = LoaderTSdf(path=path, datatype=datatype)
        # Simple model
        cnst_model = TSmodels.Constant(dim=dim, rg=rg, feature_label=feature_label)
        # Generate
        cnst_model.generate(N)
        ID = str(cnst_model)
        cnst_model.register_data(loader)

        # Test Forecast
        # forecast IS
        timeseries = loader.get_timeseries(IDs=[ID], end=[3], features=["feature"])
        cnst_model.set_data(timeseries)
        cnst_model.forecast(T, reset_timestamp=False, collision="overwrite")
        cnst_model.register_data(loader, append_to_feature=str(cnst_model))
        ## forecast OOS
        timeseries = loader.get_timeseries(IDs=[ID], end_index=10, features=["feature"])
        cnst_model.set_data(timeseries)
        cnst_model.forecast(T, reset_timestamp=False, collision="overwrite")
        cnst_model.register_data(loader, append_to_feature=str(cnst_model))

        # Test rolling forecast
        loader.rm_feature("feature_Constant")
        ## rolling forecast IS
        timeseries = loader.get_timeseries(IDs=[ID], end=[3], features=["feature"])
        cnst_model.set_data(timeseries)
        cnst_model.rolling_forecast(T)
        cnst_model.register_data(loader, append_to_feature=str(cnst_model))
        ## rolling forecast OOS
        timeseries = loader.get_timeseries(IDs=[ID], end_index=10, features=["feature"])
        cnst_model.set_data(timeseries)
        cnst_model.rolling_forecast(T)
        cnst_model.register_data(loader, append_to_feature=str(cnst_model))

        # Rolling forecast parameters
        ## batch_size > window_size
        T = 14
        batch_size = 5
        window_size = 2
        timeseries = loader.get_timeseries(
            IDs=["Constant"], start_index=0, end_index=10, features=["feature"]
        )
        cnst_model.set_data(timeseries)
        cnst_model.rolling_forecast(T, batch_size=batch_size, window_size=window_size)
        assert cnst_model.get_data().size == 10 * dim
        ### train == True
        cnst_model.set_data(timeseries)
        cnst_model.rolling_forecast(
            T, train=True, batch_size=batch_size, window_size=window_size
        )
        ## batch_size > T
        T = 7
        batch_size = 9
        window_size = 2
        timeseries = loader.get_timeseries(
            IDs=["Constant"], start_index=0, end_index=10, features=["feature"]
        )
        cnst_model.set_data(timeseries)
        cnst_model.rolling_forecast(T, batch_size=batch_size, window_size=window_size)
        assert cnst_model.get_data().size == 0
        #### side == after
        cnst_model.set_data(timeseries)
        cnst_model.rolling_forecast(
            T, batch_size=batch_size, window_size=window_size, side="after"
        )
        assert cnst_model.get_data().size == 9 * dim
        cnst_model.register_data(loader, append_to_feature="Constant_rolling")


def test_parametersIO():
    seed = 1234
    feature_label = ["feature"]
    rg = Generator(PCG64(seed))
    # loader
    # Simple model
    cnst_model = TSmodels.Constant(lag=1, dim=1, rg=rg, feature_label=feature_label)
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


@pytest.mark.arma
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


def test_GARCH():
    lag = 3
    N = 10
    rg = Generator(PCG64(1234))
    garch_model = TSmodels.GARCH(dim=1, lag=lag, rg=rg)
    timeseries = garch_model.generate(N=N)
    garch_model.set_data(data=timeseries)
    timeseries = garch_model.generate(N=N)


def test_MGARCH():
    """MGARCH test.

    Need to be careful since VEC_GARCH is not well defnied for
    negative definite matrix.
    """
    # VEC_GARCH, dim=2
    seed = 9103
    lag = 3
    N = 10
    dim = 2
    rg = Generator(PCG64(seed))
    vec_garch = TSmodels.VEC_GARCH(dim=dim, lag=lag, rg=rg)
    vec_garch.generate(N=N)

    # VEC_SPD_GARCH
    for dim in range(3, 5):
        vec_spd_garch = TSmodels.VEC_SPD_GARCH(dim=dim, lag=lag, rg=rg)

        timeseries = vec_spd_garch.generate(N=N)
        vec_spd_garch.set_data(data=timeseries)
        vec_spd_garch.generate(N=N)


@pytest.mark.R
def test_Rgarch():
    """Using R packages"""
    lag = 2
    N = 10
    T = 10
    rg = Generator(PCG64(1234))
    garch_model = TSmodels.GARCH(lag=lag, rg=rg)
    timeseries = garch_model.generate(N=N)
    rgarch_model = TSmodels.rGARCH(lag=lag, rg=rg)
    rgarch_model.set_data(data=timeseries)
    timeseries = rgarch_model.forecast(T=T)
