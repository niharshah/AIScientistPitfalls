import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    cg = experiment_data["cross_dataset_generalization"]
    methods = ["SPR_BENCH", "SPR+SHAPE+COLOR"]
    colors = {"SPR_BENCH": "tab:blue", "SPR+SHAPE+COLOR": "tab:orange"}
    h_sizes = sorted(cg.keys())

    # helper to fetch arrays
    def arr(store, key):
        return np.array(store["losses"][key])  # shape (E,2)

    def hwa(store):
        return np.array(store["metrics"]["val"])  # shape (E,4)

    # -------------- Figure 1 : hidden=256, losses + HWA --------------
    try:
        hs = 256
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for m in methods:
            train = arr(cg[hs][m], "train")
            val = arr(cg[hs][m], "val")
            hwav = hwa(cg[hs][m])
            axs[0].plot(
                train[:, 0],
                train[:, 1],
                label=f"{m} train",
                color=colors[m],
                linestyle="-",
            )
            axs[0].plot(
                val[:, 0], val[:, 1], label=f"{m} val", color=colors[m], linestyle="--"
            )
            axs[1].plot(hwav[:, 0], hwav[:, 3], label=m, color=colors[m])
        axs[0].set_title("Training vs Validation Loss")
        axs[1].set_title("Harmonic Weighted Accuracy")
        for ax in axs:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.legend()
        fig.suptitle("Hidden=256 | Left: Loss Curves, Right: HWA | Dataset: SPR_BENCH")
        fig.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_hidden256_loss_hwa.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating figure hidden256: {e}")
        plt.close()

    # -------------- Figure 2 : HWA curves for all h --------------
    try:
        plt.figure(figsize=(8, 5))
        for hs in h_sizes:
            for m, ls in zip(methods, ["-", "--"]):
                hwav = hwa(cg[hs][m])
                label = f"{m} h={hs}"
                plt.plot(
                    hwav[:, 0], hwav[:, 3], label=label, color=colors[m], linestyle=ls
                )
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("HWA vs Epoch for Each Hidden Size | Dataset: SPR_BENCH")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_all_hidden_hwa_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curve figure: {e}")
        plt.close()

    # -------------- Figure 3 : Final-epoch HWA bar chart --------------
    try:
        width = 0.35
        x = np.arange(len(h_sizes))
        hwa_final = {m: [hwa(cg[h][m])[-1, 3] for h in h_sizes] for m in methods}
        plt.figure(figsize=(8, 4))
        plt.bar(
            x - width / 2,
            hwa_final["SPR_BENCH"],
            width,
            label="SPR_BENCH",
            color=colors["SPR_BENCH"],
        )
        plt.bar(
            x + width / 2,
            hwa_final["SPR+SHAPE+COLOR"],
            width,
            label="SPR+SHAPE+COLOR",
            color=colors["SPR+SHAPE+COLOR"],
        )
        plt.xticks(x, h_sizes)
        plt.xlabel("Hidden Dimension")
        plt.ylabel("Final Epoch HWA")
        plt.title("Final Harmonic Weighted Accuracy by Hidden Size")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating final HWA bar chart: {e}")
        plt.close()
