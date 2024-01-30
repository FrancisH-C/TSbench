import numpy as np
from numpy.random import Generator, PCG64
from randomgen import Xoshiro256
from TSbench.TSdata import LoaderTSdf
from TSbench import TSmodels

path = "data/test_models"
datatype = "simulated"
loader = LoaderTSdf(path=path, datatype=datatype)

seed = 1234
ID = "Constant"
feature_label = ["feature"]
cnst_model = TSmodels.Constant(dim=1, rg=Generator(PCG64(seed)),
                               feature_label=feature_label)

T = 10
cnst_model.generate(T)
cnst_model.register_data(loader, collision="overwrite")

timeseries = loader.get_timeseries(IDs=["Constant"], end_index=4, features=["feature"])
cnst_model.set_data(data=timeseries)
cnst_model.train()

timeseries = loader.get_timeseries(IDs=["Constant"], end=3, features=["feature"])
cnst_model.set_data(data=timeseries)
T = 5
cnst_model.forecast(T)
cnst_model.register_data(loader, append_to_feature=str(cnst_model))

timeseries = loader.get_timeseries(IDs=["Constant"], end_index=10, features=["feature"])
cnst_model.set_data(data=timeseries)
T = 5
cnst_model.forecast(T)
cnst_model.register_data(loader, append_to_feature=str(cnst_model))

loader.write()
