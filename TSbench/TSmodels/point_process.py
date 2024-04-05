from typing import Optional

from abc import abstractmethod
import numpy as np
from numpy.random import PCG64, Generator


class PointProcess:
    current_timestamp: int
    timestamp: np.ndarray
    rg: Generator

    def __init__(
        self,
        current_timestamp: int = 0,
        rg: Optional[Generator] = None,
    ) -> None:
        if rg is None:
            rg = Generator(PCG64())

        self.rg = rg
        self.set_current_timestamp(current_timestamp)

    def __iter__(self):
        return self

    def __next__(self):
        self.current_timestamp += 1
        return self.current_timestamp

    def get_current_timestamp(self) -> int:
        return self.current_timestamp

    def set_current_timestamp(
        self, current_timestamp: Optional[int] = None, next_step: bool = False
    ):
        if isinstance(current_timestamp, list) and len(current_timestamp) == 1:
            current_timestamp = current_timestamp[0]

        if current_timestamp is None:
            current_timestamp = 0

        self.current_timestamp = current_timestamp
        if next_step:
            next(self)

    @abstractmethod
    def generate_timestamp(self, nb_points: int = 0) -> np.ndarray:
        pass


class Deterministic(PointProcess):
    def __init__(self, **pp_args) -> None:
        super().__init__(**pp_args)

    def generate_timestamp(self, nb_points: int = 0) -> np.ndarray:
        self.timestamp = np.array(
            range(self.current_timestamp, self.current_timestamp + nb_points)
        )
        return self.timestamp
