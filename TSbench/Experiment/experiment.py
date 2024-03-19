"""Experiment module defing the class."""
# from TSload import TSloader
import pandas as pd
from numpy.random import Generator
from TSbench.TSdata.TSloader import LoaderTSdf
from typing import Callable
from joblib import Parallel, delayed
from randomgen import Xoshiro256


class ExperimentTest:
    """A collection of loaders and a function to apply to them using multiprocessing.


    Need to respect:
    - n_procs + n_loaders <= n threads
    - 2 * n_loaders <= n threads

    Args:
        loaders (TSLoader): Sets attribute of the same name.
        function Callable[[TSloader], None]): Sets attribute of the same name.

    Attributes:
        loaders ("TSLoader"): list of loaders to use with their split_pattern for
            multiprocessing.
        df_function Callable[["TSloader"], None]): A function to apply to every a df
            every loaders.
        loader_function Callable[["TSloader"], None]): A function to apply to every split_pattern of
            every loaders.

    """

    def __init__(
        self,
        data_path=None,
        output_path=None,
        input_datatype="",
        output_datatype=None,
        n_procs=1,
        n_loaders=1,
        IDs=None,
        subsplit_pattern=None,
        loader_function: Callable[["TSloader"], None] = None,
        df_function: Callable[[pd.DataFrame], None] = None,
        loaders: "TSloader" = None,
        autoload=True,
        parallel=True,
    ):
        """Init method."""

        def df_function_wrap(loader, IDs):
            if type(IDs) is not list:
                IDs = [IDs]
            try:
                df = loader.get_df(IDs=IDs)
            except KeyError:
                print(IDs, "not in ", loader.current_split, "data")
                return

            split = loader.current_split + "_" + "".join(IDs)
            output_loader = LoaderTSdf(
                path=self.output_path,
                datatype=self.output_datatype,
                split_pattern=split,
                parallel=self.parallel,
            )
            df_function(df, output_loader)

        self.set_loaders(
            data_path=data_path,
            datatype=intput_datatype,
            subsplit_pattern=subsplit_pattern,
            n_loaders=n_loaders,
            loaders=loaders,
        )

        self.datatype = self.loaders[0].datatype
        self.data_path = self.loaders[0].path

        if output_path is None:
            output_path = self.data_path
        if output_datatype is None:
            output_datatype = self.datatype

        if loader_function is None:
            loader_function = lambda loader: None
        if df_function is None:
            df_function = lambda split, ID, df: None

        if IDs is None:
            IDs = self.loaders[0].get_IDs()

        self.loader_function = loader_function
        self.df_function = df_function_wrap
        self.n_procs = n_procs
        self.output_path = output_path
        self.output_datatype = output_datatype
        self.IDs = IDs
        self.autoload = autoload
        self.parallel = parallel

    def run_ID(self, merge_data=False):
        """For every ID, apply `df_function` attribute optionally in parallel."""

        def df_ID_function(loader):
            for split in loader.subsplit_pattern:
                loader.set_current_split(split, autoload=self.autoload)
                # run df_function in parallel with the available n_procs
                if self.parallel:
                    Parallel(n_jobs=self.n_procs)(
                        delayed(self.df_function)(loader, IDs) for IDs in self.IDs
                    )
                else:
                    for IDs in self.IDs:
                        self.df_function(loader, IDs)

        if self.parallel:
            Parallel(n_jobs=len(self.loaders))(
                delayed(df_ID_function)(loader) for loader in self.loaders
            )
        else:
            for loader in self.loaders:
                df_ID_function(loader)

        if self.parallel and merge_data:
            output_loader = LoaderTSdf(
                path=self.output_path, datatype=self.output_datatype
            )
            output_loader.merge_metadata()
            LoaderTSdf.merge_splitted_files(output_loader, len(self.loaders))

    def run_loader(self):
        def loaders_split_function(loader):
            for split in loader.subsplit_pattern:
                loader.set_current_split(split, autoload=self.autoload)
                self.loader_function(loader)

        if self.parallel:
            Parallel(n_jobs=len(self.loaders))(
                delayed(loaders_split_function)(loader) for loader in self.loaders
            )
        else:
            for loader in self.loaders:
                loaders_split_function(loader)

    def set_loaders(
        self,
        data_path=None,
        datatype="",
        subsplit_pattern=None,
        n_loaders=1,
        loaders: "TSloader" = None,
    ):
        if loaders is None:
            if data_path is None:
                raise ValueError("Need data_path")
            metaloader = LoaderTSdf(data_path, datatype)

            if subsplit_pattern is None:
                subsplit_pattern = metaloader.get_split_pattern()
            last_split_index = len(subsplit_pattern)

            if last_split_index == 0:
                raise ValueError("split_pattern not defined.")
            if n_loaders > last_split_index:
                raise ValueError("n_loaders greater than length of subsplit pattern.")

            loaders = []
            for i in range(n_loaders - 1):
                subsplit_index = slice(
                    *[last_split_index // n_loaders * j for j in [i, i + 1]]
                )
                loader = LoaderTSdf(
                    data_path,
                    datatype,
                    subsplit_pattern=subsplit_pattern[subsplit_index],
                    autoload=False,
                )
                loaders.append(loader)
            # i == n_loaders
            subsplit_index = slice(
                last_split_index // n_loaders * (n_loaders - 1), last_split_index
            )
            loader = LoaderTSdf(
                data_path,
                datatype,
                subsplit_pattern=subsplit_pattern[subsplit_index],
                autoload=False,
            )
            loaders.append(loader)

        self.loaders = loaders


class ExperimentTest2:
    """."""

    def __init__(
        self,
        data_loader=None,
        process_functions: list[Callable[[pd.DataFrame], [pd.DataFrame]]] = None,
        output_loader=None,
        rg: Generator = None,
    ) -> None:
        """Exeperiment."""

        if data_loader is None:
            data_loader = LoaderTSdf(datatype="experiment")
        if output_loader is None:
            output_loader = LoaderTSdf(datatype="experiment")
        self.data_loader = data_loader
        self.output_loader = output_loader

        if rg is None:
            rg = Generator(Xoshiro256())
        self.rg = rg

        self.process_functions = process_functions

    def run_experiment(self, write=True):
        """."""
        for function in self.process_functions:
            # load data with self.data_lodaer
            data = self.data_loader.get_timeseries()
            # add the output of function to self.output_loader
            self.output_loader.add_data(function(data))
            if write:
                # write the output
                self.output_loader.write()


class ExperimentTest3:
    """."""

    def __init__(
        self,
        data_loader=None,
        output_loader=None,
        rg: Generator = None,
        run_functions: list[Callable[[pd.DataFrame], [pd.DataFrame]]] = None,
        n_procs=1,
        n_loaders=1,
        parallel=False,
    ) -> None:
        """."""

        if data_loader is None:
            data_loader = LoaderTSdf(datatype="experiment")
        if output_loader is None:
            output_loader = LoaderTSdf(datatype="experiment_output")
        self.data_loader = data_loader
        self.output_loader = output_loader

        if rg is None:
            rg = Generator(Xoshiro256())
        self.rg = rg

        self.run_functions = run_functions

        self.parallel = parallel

    def run_experiment(self, write=False):
        """."""
        for function in self.run_functions:
            # load data with self.data_lodaer
            data = self.data_loader.get_timeseries()
            # add the output of function to self.output_loader
            self.output_loader.add_data(function(data))
            if write:
                # write the output
                self.output_loader.write()

    def save(self):
        loader = self.loaders[0]
        for generator in self.generator_models:
            generator.generate(100)
            # df = pd.DataFrame.from_dict(data=generator.outputs)
            # loader.add_data()

    def load(self):
        for loader in self.loaders:
            loader.load()


class SimpleExperiment:
    def __init__(
        self,
        data_path=None,
        output_path=None,
        input_datatype="",
        output_datatype=None,
        n_procs=1,
        n_loaders=1,
        IDs=None,
        subsplit_pattern=None,
        loader_function: Callable[["TSloader"], None] = None,
        df_function: Callable[[pd.DataFrame], None] = None,
        loaders: "TSloader" = None,
        autoload=True,
        parallel=True,
    ):
        super().__init__()

    def run_experiment(self, fn_models):
        """."""
        i = 0

        for model in self.models:
            fm_models[i](model)
            # write the output with self.output_loader
            # generator.generate(self.N)
            i += 1


class Experiment:
    def __init__(
        self,
        data=None,
        generator_models=None,
        forecast_models=None,
        output_function=None,
        rg: Generator = None,
        N=None,
        T=None,
        rolling_window_size=None,
    ):
        self.update_expriment(
            run,
            data=data,
            generator_models=generator_models,
            forecast_models=forecast_models,
            output_function=output_function,
            rg=rg,
            N=N,
            T=T,
            rolling_window_size=rolling_window_size,
        )

    def update_experiment(
        run=True,
        data=None,
        generator_models=None,
        forecast_models=None,
        output_function=None,
        rg: Generator = None,
        N=None,
        T=None,
        rolling_window_size=None,
    ):
        if data is not None:
            self.data = data
            self.generator_models = generator_models
            self.forecast_models = forecast_models
            self.rg = rg

    def run_experiment(self, fn_models):
        """."""
        i = 0

        for model in self.models:
            fm_models[i](model)
            # write the output with self.output_loader
            # generator.generate(self.N)
            i += 1
