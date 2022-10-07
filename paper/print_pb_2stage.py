import lib
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("out", exist_ok=True)

def get_fig(sweepname):
    KEY = "validation/test/accuracy/test_group_alternate"

    runs = lib.get_runs([sweepname])
    g = lib.common.group(runs, ['fit.overlap', 'fit.symbol_overlap'])
    s = lib.common.calc_stat(g, lambda k: k in {KEY})

    overlaps = {}
    for n, st in s.items():
        ov, sov = n.split("/")
        overlap = int(ov.split("_")[-1])
        symbol_overlap = int(sov.split("_")[-1])

        overlaps[overlap, symbol_overlap] = st[KEY].get()

    all_overlaps = list(sorted(set(n[0] for n in overlaps.keys())))
    all_symbol_overlaps = list(sorted(set(n[1] for n in overlaps.keys())))

    #print(overlaps)

    means = np.zeros([len(all_overlaps), len(all_symbol_overlaps)], dtype=np.float32)
    stds = np.zeros([len(all_overlaps), len(all_symbol_overlaps)], dtype=np.float32)

    for i, o in enumerate(all_overlaps):
        for j, s in enumerate(all_symbol_overlaps):
            means[i, j] = overlaps[o, s].mean
            stds[i, j] = overlaps[o, s].std


    figure = plt.figure(figsize=[4,2])#means.shape)

    im = plt.imshow(means, interpolation='nearest', cmap=plt.cm.viridis, aspect='auto')

    for i in range(len(all_overlaps)):
        for j in range(len(all_symbol_overlaps)):
            color = "white" if means[i, j] < 0.5 else "black"
            plt.text(j, i, f"${means[i, j]:.2f} \\pm {stds[i, j]:.1f}$", ha="center", va="center", color=color, fontsize=8)


    plt.yticks(range(len(all_overlaps)), [str(a) for a in all_overlaps])
    plt.xticks(range(len(all_symbol_overlaps)), [str(a) for a in all_symbol_overlaps])

    plt.xlabel("No. of symbols/function")
    plt.ylabel("No. of functions")
    return figure


get_fig("fit_parallel_branches_2stage_overlap_ndr").savefig(f"out/twostage.pdf", bbox_inches='tight', pad_inches = 0.01)
get_fig("fit_parallel_branches_2stage_overlap_trafo").savefig(f"out/twostage_trafo.pdf", bbox_inches='tight', pad_inches = 0.01)
get_fig("fit_parallel_branches_2stage_overlap_rnn").savefig(f"out/twostage_rnn.pdf", bbox_inches='tight', pad_inches = 0.01)
