import lib
import torch
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools

os.makedirs("out", exist_ok=True)

def plot(f: str, y_transform):
    err = torch.load(f)

    figure, (ax1, cax) = plt.subplots(ncols=2, figsize=[1,0.8], gridspec_kw={"width_ratios":[10,1]})
    # figure = plt.figure()#means.shape)

    im = ax1.imshow(err["map"], interpolation='nearest', cmap=plt.cm.viridis, aspect='auto', vmin=0, vmax=1)


    plt.sca(ax1)
    for i, j in itertools.product(range(err["map"].shape[0]), range(err["map"].shape[1])):
        color = "white" if err["map"][i, j] < 0.5 else "black"
        plt.text(j, i, f"${err['map'][i, j]:.2f}$", ha="center", va="center", color=color, fontsize=8)

    plt.yticks(range(len(err["y_marks"])), [y_transform(c) for c in err["y_marks"]], fontsize=8)
    plt.xticks(range(len(err["x_marks"])), [f"$G_{c}$" for c in err["x_marks"]], fontsize=8) #rotation=45, fontsize=8, ha="right", rotation_mode="anchor")

    o = 0.5
    ax1.plot([1-o, 1-o], [0-o, 2-o], color="black", linewidth=1)
    ax1.plot([0-o, 2-o], [1-o, 1-o], color="black", linewidth=1)

    # offsets = [0]
    cbar = plt.colorbar(im, cax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(8)

    plt.subplots_adjust(wspace=0.1)

    return figure

plot("exported_files/raw_plots/representation_cos_distance_cluster_error/symbol_3.pth", lambda x: f"$C_{int(x)+1}$").savefig("out/cluster_error_sym3.pdf", bbox_inches='tight', pad_inches = 0.01)
plot("exported_files/raw_plots/representation_cos_distance_group_error/symbol_3.pth", lambda x: f"$G_{x}$").savefig("out/group_error_sym3.pdf", bbox_inches='tight', pad_inches = 0.01)
