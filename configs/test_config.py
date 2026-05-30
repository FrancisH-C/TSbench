from TSbench.TSdata.TSloader import LoaderTSdf
from numpy.random import Generator, PCG64
from TSbench.TSmodels import Constant, ARMA
from typing import Any


###########################
# Configuration : General #
###########################
# paths
initialize_path = "data/test_models/"  # path to unprocessed data
data_path = "data/test_experiment/"
output_path = "data/test_experiment/"

# data options
datatype = "simulated"

# parallel options
n_jobs = 1
n_input_loaders = 1

# Forecast models
# Here, using the same as Generate models


##############################
# Configuration : Initialize #
##############################
def initialize_function():
    inititalize_loader = LoaderTSdf(datatype=datatype, path=data_path)
    inititalize_loader.restart_dataset()


#######################
# Configuration : run #
#######################
# Generate models
feature_label = ["feature"]
dim_label = ["first"]

cnst_model = Constant(
    rg=Generator(PCG64(1234)), dim_label=dim_label, feature_label=feature_label
)
arma_model = ARMA(
    lag=1, rg=Generator(PCG64(4321)), dim_label=dim_label, feature_label=feature_label
)

# Forecast models


############################
# Configuration : Output #
############################
def output_process_split(_input_loader, _ouptut_loader):
    return


#####################
# Set Configuration #
#####################
general: dict[str, Any] = {
    "data_path": data_path,
    "output_path": output_path,
    "datatype": datatype,
    "n_jobs": n_jobs,
    "n_input_loaders": n_input_loaders,
}
initialize: dict[str, Any] = {"function": initialize_function}
pre_process: dict[str, Any] = {}
run_process: dict[str, Any] = {}
generate: dict[str, Any] = {
    "models": [cnst_model, arma_model],
    "params": {"N": 10},
}
train: dict[str, Any] = {
    "models": [cnst_model, arma_model],
}
forecast: dict[str, Any] = {
    "models": [arma_model],
    "params": {"T": 10},
}
output: dict[str, Any] = {"process_split": output_process_split}
