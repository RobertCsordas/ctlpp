import lib
import torch
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

os.makedirs("out", exist_ok=True)

def plot_core(ax, p):
    im = ax.imshow(p["map"], interpolation='nearest', cmap=plt.cm.viridis, aspect='auto', vmin=0, vmax=1)

    plt.sca(ax)
    plt.yticks(range(len(p["y_marks"])), [x[1:] for x in p["y_marks"]], fontsize=8)
    plt.xticks(range(len(p["x_marks"])), [x[1:] for x in p["x_marks"]], fontsize=8) #rotation=45, fontsize=8, ha="right", rotation_mode="anchor")

    offsets = [0]

    grp = [x[0] for x in p["x_marks"]]
    grpnames = []
    for i, g in enumerate(grp[1:]):
        if g != grp[i] or i == len(grp) - 2:
            offsets.append(i)
            grpnames.append(grp[i])

    labelpos = []
    for i in range(1, len(offsets)):
        labelpos.append((offsets[i] + offsets[i - 1]) / 2)

    # return figure
    ax2 = ax.twiny()

    ax2.spines["bottom"].set_position(("axes", -0.075))
    ax2.tick_params('both', length=0, width=0, which='minor', labelsize=8)
    ax2.tick_params('both', direction='in', which='major', labelsize=8)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    ax2.set_xticks(offsets)
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(ticker.FixedLocator(labelpos))
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter([f"$G_{n}$" for n in grpnames]))

    # return figure

    ax2 = ax.twinx()

    ax2.spines["left"].set_position(("axes", -0.07))
    ax2.tick_params('both', length=0, width=0, which='minor', labelsize=8)
    ax2.tick_params('both', direction='in', which='major', labelsize=8)
    ax2.yaxis.set_ticks_position("left")
    ax2.yaxis.set_label_position("left")

    maxy = len(p["y_marks"]) - 1
    ax2.set_yticks([maxy - y - 1 for y in offsets])
    ax2.yaxis.set_major_formatter(ticker.NullFormatter())
    ax2.yaxis.set_minor_locator(ticker.FixedLocator([maxy - y - 1 for y in labelpos]))
    ax2.yaxis.set_minor_formatter(ticker.FixedFormatter([f"$G_{n}$" for n in grpnames]))

    return im


def plot(f:str, write_cluster_labels: bool = False, higlight_clusters: bool = True):
    p = torch.load(f)

    # print(p)

    figure, (ax1, cax) = plt.subplots(ncols=2, figsize=[4.5,4], gridspec_kw={"width_ratios":[20,1]})
    # figure = plt.figure()#means.shape)

    im = plot_core(ax1, p)

    n_clusters = p["idx_to_cluster"].max()
    cluster_offsets = np.cumsum([(p["idx_to_cluster"] == (c+1)).sum() for c in range(n_clusters)]).tolist()
    cluster_offsets = [0] + cluster_offsets

    plt.sca(ax1)
    if higlight_clusters:
        # marker_color = "#FA9500"
        marker_color = "#E15554"
        # marker_color = "black"
        # marker_color = "white"
        linewidth = 2
        for i in range(1, len(cluster_offsets)):
            # bottom
            ax1.plot([cluster_offsets[i-1]-0.5, cluster_offsets[i]-0.5], [cluster_offsets[i]-0.5, cluster_offsets[i]-0.5], color=marker_color, linewidth=linewidth)

            # top
            ax1.plot([cluster_offsets[i-1]-0.5, cluster_offsets[i]-0.5], [cluster_offsets[i-1]-0.5, cluster_offsets[i-1]-0.5], color=marker_color, linewidth=linewidth)

            #right
            ax1.plot([cluster_offsets[i]-0.5, cluster_offsets[i]-0.5], [cluster_offsets[i-1]-0.5, cluster_offsets[i]-0.5], color=marker_color, linewidth=linewidth)

            #left
            ax1.plot([cluster_offsets[i-1]-0.5, cluster_offsets[i-1]-0.5], [cluster_offsets[i-1]-0.5, cluster_offsets[i]-0.5], color=marker_color, linewidth=linewidth)

        if write_cluster_labels:
            for i in range(1, len(cluster_offsets)):
                xy = (cluster_offsets[i-1] + cluster_offsets[i]) / 2 - 0.5
                plt.text(xy, xy, f"$C_{i}$", ha="center", va="center", color=marker_color, fontsize=16)

    # divider = make_axes_locatable(ax1)
    # ax2 = ax1.twinx()
    # ax2.spines["left"].set_position(("axes", -0.07))
    # ax2.tick_params('both', length=0, width=0, which='minor', labelsize=8)
    # ax2.tick_params('both', direction='in', which='major', labelsize=8)
    # cax = divider.append_axes("right", size=0.25, pad=0.1)
    cbar = plt.colorbar(im, cax)
    for i, t in enumerate(cbar.ax.get_yticklabels()):
        t.set_fontsize(8)
    
    plt.subplots_adjust(wspace=0.06)

    return figure

figure = plot("exported_files/raw_plots/representation_cos_distance/symbol_3.pth", write_cluster_labels=True)
figure.savefig("out/cossim_3.pdf", bbox_inches='tight', pad_inches = 0.01)

figure = plot("exported_files/raw_plots/representation_cos_distance/symbol_6.pth")
figure.savefig("out/cossim_6.pdf", bbox_inches='tight', pad_inches = 0.01)