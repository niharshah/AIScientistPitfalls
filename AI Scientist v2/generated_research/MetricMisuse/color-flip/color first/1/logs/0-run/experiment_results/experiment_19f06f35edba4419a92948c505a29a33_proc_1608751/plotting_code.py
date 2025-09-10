import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- setup -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

tags = list(experiment_data.keys())
if not tags:
    print("No experiment data found, nothing to plot.")
else:
    # gather summaries --------------------------------------------------------
    epochs_dict, train_loss, val_loss, val_cva = {}, {}, {}, {}
    test_metrics = {}  # tag -> dict
    for tag in tags:
        ed = experiment_data[tag]["SPR_BENCH"]
        train_loss[tag] = ed["losses"]["train"]
        val_loss[tag] = ed["losses"]["val"]
        val_cva[tag] = [m["cva"] for m in ed["metrics"]["val"]]
        test_metrics[tag] = ed["metrics"]["test"]
        epochs_dict[tag] = list(range(1, len(train_loss[tag]) + 1))

    # ---------------- plot 1 : Loss curves -----------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for tag in tags:
            axes[0].plot(epochs_dict[tag], train_loss[tag], label=tag)
            axes[1].plot(epochs_dict[tag], val_loss[tag], label=tag)
        axes[0].set_title("Train Loss")
        axes[1].set_title("Validation Loss")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-Entropy")
            ax.legend()
        fig.suptitle("SPR_BENCH Loss Curves (Left: Train, Right: Validation)")
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------------- plot 2 : Validation CVA curves -------------------------
    try:
        plt.figure(figsize=(6, 4))
        for tag in tags:
            plt.plot(epochs_dict[tag], val_cva[tag], label=tag)
        plt.title("SPR_BENCH Validation Composite Variety Accuracy (CVA)")
        plt.xlabel("Epoch")
        plt.ylabel("CVA")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_CVA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating validation CVA plot: {e}")
        plt.close()

    # ---------------- plot 3 : Test CVA bar chart ----------------------------
    try:
        plt.figure(figsize=(6, 4))
        names, scores = zip(*[(t, test_metrics[t]["cva"]) for t in tags])
        plt.bar(names, scores, color="skyblue")
        plt.title("SPR_BENCH Test CVA by Model Variant")
        plt.ylabel("CVA")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_CVA_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test CVA bar plot: {e}")
        plt.close()

    # ---------------- plot 4 : CWA vs SWA scatter ----------------------------
    try:
        plt.figure(figsize=(6, 5))
        cwa_vals = [test_metrics[t]["cwa"] for t in tags]
        swa_vals = [test_metrics[t]["swa"] for t in tags]
        plt.scatter(cwa_vals, swa_vals, c="orange")
        for i, t in enumerate(tags):
            plt.text(cwa_vals[i], swa_vals[i], t, fontsize=8, ha="left", va="bottom")
        plt.title("SPR_BENCH Test CWA vs SWA (Dataset: SPR_BENCH)")
        plt.xlabel("Color-Weighted Acc.")
        plt.ylabel("Shape-Weighted Acc.")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_CWA_vs_SWA_scatter.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CWA vs SWA scatter plot: {e}")
        plt.close()

    # -------------- print final test metrics ---------------------------------
    print("\nTest-set performance:")
    for tag in tags:
        met = test_metrics[tag]
        print(
            f"{tag}: CWA={met['cwa']:.4f}, SWA={met['swa']:.4f}, CVA={met['cva']:.4f}"
        )
