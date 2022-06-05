from torchaudio.functional import edit_distance

from common.tool.CTC import CTC, CTC_label


class CER:
    def __init__(self, blank=0, reduction="mean"):
        self.ctc_prob = CTC(blank)
        self.ctc_lab = CTC_label(blank)
        if reduction == "mean":
            self.mean = 1
        elif reduction == "none":
            self.mean = 0
        else:
            assert 0, "reduction ha mean ka none ni shitekudasai"

    def __call__(self, prob, lab):
        """
        prob: FloatTensor[batch,sequence,class]
        lab: LongTensor[batch,sequence]
        """
        cer = 0.0
        for prob_lab_h_i, lab_i in zip(prob, lab):
            lab_h_i = self.ctc_prob(prob_lab_h_i)
            lab_i = self.ctc_lab(lab_i)
            cer += edit_distance(lab_h_i, lab_i) / len(lab_i)
        if self.mean:
            cer /= lab.size(0)
        return cer
