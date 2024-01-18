import numpy as np

class PointProcess():
    def __init__(
        self,
        current_timestamp = 0,
        rg = None,
        timestamp_style = "incr",
    ):
        self.rg = rg
        self.timestamp_style = timestamp_style
        self.set_current_timestamp(current_timestamp)

    def __iter__(self):
            return self

    def __next__(self):
        if self.timestamp_style == "incr":
            self.current_timestamp += 1
        return self.current_timestamp

    def get_current_timestamp(self):
        return self.current_timestamp

    def set_current_timestamp(self, current_timestamp = None, next_step=False):
        if type(current_timestamp) is list and len(current_timestamp) == 1:
            current_timestamp = current_timestamp[0]

        if current_timestamp is None:
            if self.timestamp_style == "incr":
                current_timestamp = 0
            elif self.timestamp_style == '%Y-%m-%d %X':
                current_timestamp = '2000-01-01'

        if self.timestamp_style == "incr" and not isinstance(current_timestamp, int | np.int64):
            raise ValueError("Need a int for this timestamp_style")

        self.current_timestamp = current_timestamp
        if next_step:
            next(self)


class Deterministic(PointProcess):
    def __init__(self, **pp_args) -> None:
        super().__init__(**pp_args)

    def generate_timestamp(self, nb_points=0):
        if self.timestamp_style == "incr":
            self.timestamp = list(range(self.current_timestamp, self.current_timestamp + nb_points))
        elif self.timestamp_style == '%Y-%m-%d %X':
            self.timestamp = list(pd.date_range(start=self.current_timestamp, periods=nb_points)
                                  .strftime(self.timestamp_style))
        #if nb_points == 0:
        #    self.set_current_timestamp()
        #for _ in range(nb_points):
        #    next(self)
        return self.timestamp


