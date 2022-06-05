import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

dpi = 200


def plot_f0(path, log2f0, fs, Nhop):
    fstart = 110
    rangeoctave = range(0, 3)

    fig, ax = plt.subplots(dpi=dpi)

    f0 = np.empty_like(log2f0)
    for i in range(len(log2f0)):
        if log2f0[i] != 0.0:
            f0[i] = log2f0[i] - np.log2(fstart)
        else:
            f0[i] = None

    time = np.arange(f0.shape[0], dtype=float)
    time = Nhop / fs * time
    ax.plot(time, f0)

    fticks = [n for n in rangeoctave]
    ax.set_yticks(fticks)
    fticklabels = [str(int(fstart * 2**n)) for n in rangeoctave]
    ax.set_yticklabels(fticklabels)
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1 / 12))

    tticks = np.arange(0, Nhop / fs * f0.shape[0], 0.5)
    ax.set_xticks(tticks)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.grid(which="major", axis="y", linewidth=0.5, color="gray")
    ax.grid(which="minor", axis="y", linewidth=0.5, color="lightgray")

    fig.savefig(path)
