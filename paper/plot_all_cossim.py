import lib
import matplotlib.pyplot as plt
import torch


def plot(dir: str):
    figure, axis = plt.subplots(ncols=4, nrows=2, figsize=[18,8])

    for sym in range(8):
        p = torch.load(f"{dir}/symbol_{sym}.pth")

        ax = axis[sym // 4, sym % 4]

        ax.set_title(f"Symbol {sym}")

        plt.sca(ax)
        im = plt.imshow(p["map"], interpolation='nearest', cmap=plt.cm.viridis, aspect='auto', vmin=0, vmax=1)
        plt.yticks(range(len(p["y_marks"])), [x[1:] for x in p["y_marks"]], fontsize=8)
        plt.xticks(range(len(p["x_marks"])), [x[1:] for x in p["x_marks"]], fontsize=8) #rotation=45, fontsize=8, ha="right", rotation_mode="anchor")

        for i, t in enumerate(ax.get_yticklabels()):
            t.set_color("red" if p["y_marks"][i][0] == "a" else "blue")

        for i, t in enumerate(ax.get_xticklabels()):
            t.set_color("red" if p["x_marks"][i][0] == "a" else "blue")

    plt.subplots_adjust(wspace=0.1)

    cbar = figure.colorbar(im, ax=axis.ravel().tolist(), pad=0.01, aspect=30)
    cbar.ax.tick_params(labelsize=8)

    return figure


plot("exported_files/raw_plots/representation_cos_distance").savefig("out/all_cossim_bad.pdf", bbox_inches='tight', pad_inches = 0.01)
plot("exported_files_good/raw_plots/representation_cos_distance").savefig("out/all_cossim_good.pdf", bbox_inches='tight', pad_inches = 0.01)
plot("exported_files_alternate/raw_plots/representation_cos_distance").savefig("out/all_cossim_alt.pdf", bbox_inches='tight', pad_inches = 0.01)
