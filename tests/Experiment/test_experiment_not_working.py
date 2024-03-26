#!/usr/bin/env python

import pytest
import numpy as np
from TSbench.TSdata import LoaderTSdf
from TSbench import TSmodels
from TSbench.Experiment import Experiment
from numpy.random import Generator, PCG64

data_loader = LoaderTSdf(path="data/test_experiment/", datatype="experiment")
output_loader = LoaderTSdf(path="data/test_experiment/", datatype="experiment")


# def test_generate_experiment():
#     def generate_constant(data):
#         N = 10
#         # Constant Model
#         cnst_model = TSmodels.Constant(dim=1)
#         # generate
#         return cnst_model.generate(N)
#
#     generate_functions = [generate_constant]
#
#     exp = Experiment(output_loader=output_loader, process_functions=generate_functions)
#
#     exp.run_experiment()
#
#
# def test_forecast_experiment():
#     def forecast_constant(data):
#         T = 10
#         # Constant Model
#         cnst_model = TSmodels.Constant(dim=1)
#         # generate
#         return cnst_model.forecast(T, data=data)
#
#     forecast_functions = [forecast_constant]
#
#     exp = Experiment(
#         data_loader=data_loader,
#         output_loader=output_loader,
#         process_functions=forecast_functions,
#     )
#
#     exp.run_experiment()
#
#
# def test_train_experiment():
#     def train_constant(df):
#         seed = 1234
#         N = 10
#         T = 5
#         # loader
#         path = "data/"
#         datatype = "simulated"
#         loader = LoaderTSdf(path=path, datatype=datatype)
#         # Constant Model
#         cnst_model = TSmodels.Constant(dim=1)
#         generator_models = [cnst_model]
#         forecasting_models = [cnst_model]
#         # generate
#         return cnst_model.generate(N)
#         # cnst_model.register_data(loader)
#         # ID = str(cnst_model)
#         # cnst_model.register_data(loader)
#         # forecast IS
#         # timeseries = loader.get_timeseries(IDs=[ID], end=3, features=["returns"])
#         # cnst_model.forecast(T, timeseries)
#         # cnst_model.register_data(loader, append_to_feature=str(cnst_model))
#         ## forecast OOS
#         # timeseries = loader.get_timeseries(IDs=[ID], end_index=10, features=["returns"])
#         # cnst_model.forecast(T, timeseries)
#         # cnst_model.register_data(loader, append_to_feature=str(cnst_model))
#
#     train_functions = [train_constant]
#
#     exp = Experiment(process_functions=train_functions)
#
#     exp.run_experiment()
