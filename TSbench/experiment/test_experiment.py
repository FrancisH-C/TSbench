from TSbench.experiment.experiment import Experiment
from configs import test_config


def test_experiment():
    experiment = Experiment(test_config)
    experiment.run(
        initialize=True,
        pre_process=True,
        generate=True,
        train=True,
        forecast=True,
        output=True,
    )


if __name__ == "__main__":
    test_experiment()
