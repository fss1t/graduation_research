import json


class Standardizer:
    def __init__(self, path_norm_in):
        # load

        with open(path_norm_in) as js:
            norm = json.loads(js.read())
        self.mean_in = norm["mean"]
        self.std_in = norm["std"]

    def __call__(self, spe):
        return spe * self.std_in + self.mean_in
