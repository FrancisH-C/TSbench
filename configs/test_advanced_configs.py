from TSbench.TSdata.TSloader import LoaderTSdf, LoadersProcess
from numpy.random import Generator, PCG64
from TSbench.TSmodels import Constant, ARMA


###########################
# Configuration : General #
###########################
# paths
pre_process_path = "data/test_models/"  # path to unprocessed data
data_path = "data/test_experiment/"
output_path = "data/test_experiment/"

# data options
datatype = "simulated"
output_datatype = "simulated"

# parallel options
n_jobs = 1
n_input_loaders = 1

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
# Here, using the same as Generate models

##############################
# Configuration : Initialize #
##############################


def initialize_function():
    inititalize_loader = LoaderTSdf(datatype=datatype, path=data_path)
    inititalize_loader.restart_dataset()


###############################
# Configuration : Pre-Process #
###############################
# pre-process functions
def pre_process_df(df):
    return df


def pre_process_split(input_loader, _ouptut_loader):
    print(input_loader.df)
    return


pre_process_output_loader = LoaderTSdf(datatype=datatype, path=data_path)
pre_process_process = LoadersProcess(
    data_path=pre_process_path,
    datatype=datatype,
    output_loader=pre_process_output_loader,
    n_jobs=n_jobs,
    n_input_loaders=n_input_loaders,
    process_df=pre_process_df,
    process_split=pre_process_split,
)

#######################
# Configuration : run #
#######################
run_output = LoaderTSdf(datatype=datatype, path=data_path)
run_process_process = LoadersProcess(
    data_path=data_path,
    datatype=datatype,
    output_loader=run_output,
    n_jobs=n_jobs,
    n_input_loaders=n_input_loaders,
    autoload=True,
)


############################
# Configuration : Output #
############################
def output_process_df(df):
    return df


def output_process_split(input_loader, _ouptut_loader):
    print(input_loader.df)
    return


output_output_loader = LoaderTSdf(datatype=datatype, path=data_path)
output_process = LoadersProcess(
    data_path=data_path,
    datatype=datatype,
    output_loader=output_output_loader,
    n_jobs=n_jobs,
    n_input_loaders=n_input_loaders,
    process_df=output_process_df,
    process_split=output_process_split,
)


#####################
# Set Configuration #
#####################

initialize = {"function": initialize_function}
pre_process = {"process": pre_process_process}
run_process = {"process": run_process_process}
generate = {
    "ID-wise": False,
    "input_loaders_params": {},
    "models": [cnst_model, arma_model],
    "params": {"N": 10},
}
train = {
    "input_loaders_params": {},
    "models": [cnst_model, arma_model],
    "params": {},
}
forecast = {
    "input_loaders_params": {},
    "models": [arma_model],
    "params": {"T": 10},
}
output = {"process": output_process, "metrics": None}
