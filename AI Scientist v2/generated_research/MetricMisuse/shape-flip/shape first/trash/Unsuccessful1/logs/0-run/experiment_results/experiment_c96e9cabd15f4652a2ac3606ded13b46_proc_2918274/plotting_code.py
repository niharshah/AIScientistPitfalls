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
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "weight_decay" in experiment_data:

    runs = experiment_data["weight_decay"]
    wd_keys = list(runs.keys())

    train_losses, val_losses, val_hwa, test_hwa = {}, {}, {}, {}

    for k in wd_keys:
        rec = runs[k]
        train_losses[k] = rec["losses"]["train"]
        val_losses[k] = rec["losses"]["val"]
        val_hwa[k] = [m["HWA"] for m in rec["metrics"]["val"]]
        test_hwa[k] = rec["metrics"]["test"]["HWA"]

    # ---------- Figure 1: Loss curves ----------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for k in wd_keys:
            axes[0].plot(train_losses[k], label=k)
            axes[1].plot(val_losses[k], label=k)
        axes[0].set_title("Left: Train Loss")
        axes[1].set_title("Right: Val Loss")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=6)
        fig.suptitle("Training and Validation Losses (SPR_BENCH)")
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- Figure 2: Validation HWA ----------
    try:
        plt.figure(figsize=(5, 4))
        for k in wd_keys:
            plt.plot(val_hwa[k], label=k)
        plt.title("Validation HWA over Epochs (SPR_BENCH)")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend(fontsize=6)
        fname = os.path.join(working_dir, "spr_bench_val_hwa_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating val HWA plot: {e}")
        plt.close()

    # ---------- Figure 3: Test HWA comparison ----------
    try:
        plt.figure(figsize=(5, 4))
        labels = list(test_hwa.keys())
        scores = [test_hwa[k] for k in labels]
        plt.bar(range(len(scores)), scores, tick_label=labels)
        for xi, sc in enumerate(scores):
            plt.text(xi, sc + 0.005, f"{sc:.3f}", ha="center", va="bottom", fontsize=7)
        plt.title("Test HWA by Weight Decay (SPR_BENCH)")
        plt.ylabel("HWA")
        fname = os.path.join(working_dir, "spr_bench_test_hwa_bar.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating test HWA bar plot: {e}")
        plt.close()

    # ---------- Console summary ----------
    print("\n=== Test HWA Summary ===")
    for k, v in test_hwa.items():
        print(f"{k:10s}: {v:.4f}")
