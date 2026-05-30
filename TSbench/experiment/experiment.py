from TSbench.TSdata.TSloader import LoaderTSdf, LoadersProcess


class Experiment:
    def __init__(self, config):
        self.set_general(config.general)
        self.set_initialize(config.initialize)
        self.set_pre_process(config.pre_process)
        self.set_run_process(config.run_process)
        self.set_generate(config.generate)
        self.set_train(config.train)
        self.set_forecast(config.forecast)
        self.set_output(config.output)

    def set_general(self, general):
        self.general = general

    def set_initialize(self, initialize):
        self.initialize = initialize

    def set_pre_process(self, pre_process):
        self.pre_process = pre_process

        # define process from other inputs
        if "process" not in self.pre_process:
            if "output_loader" not in self.pre_process:
                self.pre_process["output_loader"] = LoaderTSdf(
                    datatype=self.general["datatype"], path=self.general["data_path"]
                )
            if "process_df" not in self.pre_process:
                self.pre_process["process_df"] = None

            if "process_split" not in self.pre_process:
                self.pre_process["process_split"] = None

            pre_process_process = LoadersProcess(
                data_path=self.general["data_path"],
                datatype=self.general["datatype"],
                output_loader=self.pre_process["output_loader"],
                n_jobs=self.general["n_jobs"],
                n_input_loaders=self.general["n_input_loaders"],
                process_df=self.pre_process["process_df"],
                process_split=self.pre_process["process_split"],
            )
            self.pre_process["process"] = pre_process_process

    def set_run_process(self, run_process):
        self.run_process = run_process

        # define process from other inputs
        if "process" not in self.run_process:
            if "output_loader" not in self.run_process:
                self.run_process["output_loader"] = LoaderTSdf(
                    datatype=self.general["datatype"], path=self.general["data_path"]
                )
            if "process_df" not in self.run_process:
                self.run_process["process_df"] = None

            if "process_split" not in self.run_process:
                self.run_process["process_split"] = None

            run_process_process = LoadersProcess(
                data_path=self.general["data_path"],
                datatype=self.general["datatype"],
                output_loader=self.run_process["output_loader"],
                n_jobs=self.general["n_jobs"],
                n_input_loaders=self.general["n_input_loaders"],
                process_df=self.run_process["process_df"],
                process_split=self.run_process["process_split"],
            )
            self.run_process["process"] = run_process_process

    def set_generate(self, generate):
        self.generate = generate
        if "input_loaders_params" not in self.generate:
            self.generate["input_loaders_params"] = {}
        if "ID-wise" not in self.generate:
            self.generate["ID-wise"] = False

    def set_train(self, train):
        self.train = train
        if "params" not in self.train:
            self.train["params"] = {}
        if "input_loaders_params" not in self.train:
            self.train["input_loaders_params"] = {}

    def set_forecast(self, forecast):
        self.forecast = forecast
        if "input_loaders_params" not in self.forecast:
            self.forecast["input_loaders_params"] = {}

    def set_output(self, output):
        self.output = output

        # define process from other inputs
        if "process" not in self.output:
            if "output_loader" not in self.output:
                self.output["output_loader"] = LoaderTSdf(
                    datatype=self.general["datatype"], path=self.general["output_path"]
                )
            if "process_df" not in self.output:
                self.output["process_df"] = None

            if "process_split" not in self.output:
                self.output["process_split"] = None

            output_process = LoadersProcess(
                data_path=self.general["data_path"],
                datatype=self.general["datatype"],
                output_loader=self.output["output_loader"],
                n_jobs=self.general["n_jobs"],
                n_input_loaders=self.general["n_input_loaders"],
                process_df=self.output["process_df"],
                process_split=self.output["process_split"],
            )
            self.output["process"] = output_process

    def configure_run_models(self, generate, train, forecast, write=True):
        def run_models(input_loader, output_loader):
            if generate:
                for model in self.generate["models"]:
                    if self.generate["ID-wise"]:
                        for ID in input_loader.get_IDs():
                            df = input_loader.get_df(
                                IDs=ID, **self.generate["input_loaders_params"]
                            )
                            model.set_data(df)
                            model.generate(**self.generate["params"])
                            # register data to use generated below in train/forecast
                            model.register_data(input_loader)
                            # register data to output
                            model.register_data(output_loader)
                    else:
                        df = input_loader.get_df(
                            **self.generate["input_loaders_params"]
                        )
                        model.set_data(df)
                        model.generate(**self.generate["params"])
                        # register data to use generated below in train/forecast
                        model.register_data(input_loader)
                        # register data to output
                        model.register_data(output_loader)

            # train or forecast
            for ID in input_loader.get_IDs():
                if train:
                    for model in self.train["models"]:
                        df = input_loader.get_df(
                            IDs=ID, **self.train["input_loaders_params"]
                        )
                        model.set_data(df)
                        model.train(**self.train["params"])
                if forecast:
                    for model in self.forecast["models"]:
                        df = input_loader.get_df(
                            IDs=ID, **self.forecast["input_loaders_params"]
                        )
                        model.set_data(df)
                        model.forecast(**self.forecast["params"])
                        model.register_data(
                            output_loader, append_to_feature=str(model), ID=ID
                        )
            if write:
                output_loader.write()

        return run_models

    def run(
        self,
        initialize=False,
        pre_process=False,
        generate=False,
        train=False,
        forecast=False,
        output=False,
    ):
        if initialize:
            self.initialize["function"]()

        if pre_process:
            self.pre_process["process"].run_process(write=True)

        run_models = self.configure_run_models(generate, train, forecast, write=True)
        self.run_process["process"].process_split = run_models

        # # run process
        self.run_process["process"].run_process(write=True)

        if output:
            self.output["process"].reload()
            self.output["process"].run_process(write=True)

    def get_output_loader(self):
        return self.output["output_loader"]
