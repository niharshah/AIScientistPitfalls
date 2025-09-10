import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["weight_decay"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed is not None:
    settings = np.array(ed["settings"])  # weight_decay values
    train_losses_all = ed["losses"]["train"]  # list of lists
    val_losses_all = ed["losses"]["val"]
    test_hwa = np.array(ed["metrics"]["test"])

    # ---------- Plot 1: loss curves ----------
    try:
        plt.figure(figsize=(6, 4))
        for wd, tr, va in zip(settings, train_losses_all, val_losses_all):
            epochs = np.arange(1, len(tr) + 1)
            plt.plot(epochs, tr, label=f"train wd={wd}")
            plt.plot(epochs, va, "--", label=f"val wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title("SPR_BENCH: Training/Validation Loss vs Epoch\n(Weight-decay sweep)")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_weight_decay.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss-curve plot: {e}")
        plt.close()

    # ---------- Plot 2: HWA vs weight-decay ----------
    try:
        plt.figure(figsize=(5, 3))
        plt.plot(settings, test_hwa, marker="o")
        plt.xlabel("Weight decay")
        plt.ylabel("Test HWA")
        plt.xscale("symlog")
        plt.title("SPR_BENCH: Test HWA vs Weight-decay")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_HWA_vs_weight_decay.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # ---------- print best setting ----------
    best_idx = int(test_hwa.argmax())
    print(
        f"Best weight_decay={settings[best_idx]} with Test HWA={test_hwa[best_idx]:.4f}"
    )
