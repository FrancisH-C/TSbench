from TSbench.TSdata.TSloader import LoaderTSdf, LoadersProcess

###########################
# Configuration : General #
###########################
# paths
pre_process_path = "data/"  # path to unprocessed data
data_path = "data/"
output_path = "data/"

# data options
datatype = "simulated"
output_datatype = "simulated"

# parallel options
n_jobs = 1
n_input_loaders = 1

# Generate models

# Forecast models

# Metrics


###############################
# Configuration : Pre-Process #
###############################
# pre-process functions
def pre_process_df(df):
    return df


def pre_process_split(_input_loader, _ouptut_loader):
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


def output_process_split(_input_loader, _ouptut_loader):
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

general = {
    "pre_process_path": pre_process_path,
    "data_path": data_path,
    "output_path": output_path,
    "datatype": datatype,
    "n_jobs": n_jobs,
    "n_input_loaders": n_input_loaders,
}
pre_process = {"process": pre_process_process}
run_process = {"process": run_process_process}
generate = {
    "ID-wise": False,
    "input_loaders_params": {},
    "models": [],
    "params": {},
}
train = {
    "input_loaders_params": {},
    "models": [],
    "params": {},
}
forecast = {
    "input_loaders_params": {},
    "models": [],
    "params": {},
}
output = {"process": output_process, "metrics": None}
