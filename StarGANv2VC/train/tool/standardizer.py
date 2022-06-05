import json


class Standardizer:
    def __init__(self, path_norm_in, path_norm_out):
        # load

        with open(path_norm_in) as js:
            norm = json.loads(js.read())
        self.mean_in = norm["mean"]
        self.std_in = norm["std"]

        with open(path_norm_out) as js:
            norm = json.loads(js.read())
        self.mean_out = norm["mean"]
        self.std_out = norm["std"]

    def __call__(self, spe):
        return (spe * self.std_in + self.mean_in - self.mean_out) / self.std_out
