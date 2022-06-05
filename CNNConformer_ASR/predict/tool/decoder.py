import json
import torch

from common.tool.CTC import CTC


class CTCDecoder:
    def __init__(self, path_phoneme, blank=0):
        self.ctc = CTC(blank)

        with open(path_phoneme, "r") as js:
            dict_phoneme = json.loads(js.read())
        self.inverse_dict = {code: phoneme for phoneme, code in dict_phoneme.items()}

    def __call__(self, prob):
        lab = self.ctc(prob.squeeze(0))
        sentence = [self.inverse_dict[code] for code in lab.tolist()]
        sentence = " ".join(sentence)
        return sentence
