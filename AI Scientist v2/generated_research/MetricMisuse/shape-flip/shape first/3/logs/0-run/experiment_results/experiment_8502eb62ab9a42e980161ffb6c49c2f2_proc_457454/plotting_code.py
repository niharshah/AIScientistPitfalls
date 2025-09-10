import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- paths ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["weight_decay"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    decays = exp["decay_values"]
    tr_losses = exp["losses"]["train"]  # list[len(decays)][epochs]
    val_losses = exp["losses"]["val"]
    val_metrics = exp["metrics"]["val"]  # list of list of dicts
    best_val_hwa = exp["best_val_hwa"]
    test_metrics = exp["test_metrics"][0] if exp["test_metrics"] else {}
    epochs = range(1, len(tr_losses[0]) + 1)

    # ----------- 1. loss curves ----------
    try:
        plt.figure(figsize=(6, 4))
        for i, wd in enumerate(decays):
            plt.plot(epochs, tr_losses[i], label=f"train wd={wd}")
            plt.plot(epochs, val_losses[i], linestyle="--", label=f"val wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR_BENCH: Training vs Validation Loss\nLeft: Train, Right: Val (per weight decay)"
        )
        plt.legend(fontsize=7)
        file_path = os.path.join(working_dir, "SPR_BENCH_loss_curves_weight_decay.png")
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ----------- 2. HWA curves -----------
    try:
        plt.figure(figsize=(6, 4))
        for i, wd in enumerate(decays):
            hwa = [m["HWA"] for m in val_metrics[i]]
            plt.plot(epochs, hwa, label=f"wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Acc")
        plt.title("SPR_BENCH: Dev HWA over Epochs\nLines: Different weight decays")
        plt.legend(fontsize=7)
        file_path = os.path.join(working_dir, "SPR_BENCH_val_HWA_curves.png")
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curve plot: {e}")
        plt.close()

    # ----------- 3. final dev HWA bar ----
    try:
        plt.figure(figsize=(5, 3))
        plt.bar(range(len(decays)), best_val_hwa, tick_label=[str(d) for d in decays])
        plt.ylabel("Final Dev HWA")
        plt.xlabel("Weight Decay")
        plt.title(
            "SPR_BENCH: Final Dev HWA per Weight Decay\nBar height = last-epoch HWA"
        )
        file_path = os.path.join(working_dir, "SPR_BENCH_final_dev_HWA_bar.png")
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating dev HWA bar plot: {e}")
        plt.close()

    # ----------- 4. test metrics bar -----
    try:
        if test_metrics:
            names = ["SWA", "CWA", "HWA"]
            vals = [test_metrics[k] for k in names]
            plt.figure(figsize=(4, 3))
            plt.bar(names, vals, color=["tab:blue", "tab:orange", "tab:green"])
            plt.ylim(0, 1)
            plt.title("SPR_BENCH: Test Metrics (Best Model)\nLeft→Right: SWA, CWA, HWA")
            file_path = os.path.join(working_dir, "SPR_BENCH_test_metrics_bar.png")
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()

    # ----------- console summary ----------
    try:
        best_idx = int(np.argmax(best_val_hwa))
        print(f"Best decay={decays[best_idx]} | Dev HWA={best_val_hwa[best_idx]:.4f}")
        if test_metrics:
            print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})
    except Exception as e:
        print(f"Error printing summary: {e}")
