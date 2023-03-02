"""Experiment module defing the class."""
#from TSload import TSloader
import os
import pandas as pd


class Experiment():
    """."""

    def __init__(
        self,
        path="experiments",
        name="experiement0",
        loaders=None,
        generator_models: str = None,
        forecasting_models: str = None,
        seed: int = None,
    ) -> None:
        """."""
        self.path = path
        self.name = name
#
#        if loaders is None:
#            loaders = [TSloader(path = os.path.join(path, "data"), datatype="simulated")]
#
#        self.loaders = loaders
#        self.generator_models = generator_models
#        self.forecasting_models = forecasting_models
#
#    def run_generator(self):
#        """."""
#        loader = self.loaders[0]
#        for generator in self.generator_models:
#            generator.generate(100)
#
#
#    def save(self):
#        loader = self.loaders[0]
#        for generator in self.generator_models:
#            generator.generate(100)
#            print(generator.outputs)
#            #df = pd.DataFrame.from_dict(data=generator.outputs)
#            print(df)
#            #loader.add_ID()
#
#    def load(self):
#        for loader in self.loaders:
#            loader.load()
#        print(loader.df)
#
#
