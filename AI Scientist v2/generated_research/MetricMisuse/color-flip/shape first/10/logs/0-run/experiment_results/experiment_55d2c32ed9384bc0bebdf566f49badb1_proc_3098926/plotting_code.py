import matplotlib.pyplot as plt
import numpy as np
import os

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


# Helper to extract metrics safely
def get_metric(metric_name):
    curves = {}
    try:
        lr_block = experiment_data["learning_rate"]["SPR_BENCH"]
        for lr, blob in lr_block.items():
            curves[lr] = blob["metrics"].get(metric_name, [])
    except Exception as e:
        print(f"Error extracting {metric_name}: {e}")
    return curves


# ---------- Figure 1: Loss Curves ----------
try:
    tr_curves = get_metric("train_loss")
    val_curves = get_metric("val_loss")
    if tr_curves and val_curves:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        axes[0].set_title("Training Loss")
        for lr, curve in tr_curves.items():
            axes[0].plot(range(1, len(curve) + 1), curve, label=f"lr={lr}")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].set_title("Validation Loss")
        for lr, curve in val_curves.items():
            axes[1].plot(range(1, len(curve) + 1), curve, label=f"lr={lr}")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        fig.suptitle(
            "SPR_BENCH Loss Curves\nLeft: Training Loss, Right: Validation Loss"
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
    else:
        print("Loss data unavailable, skipping loss plot.")
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- Figure 2: Weighted Accuracy Curves ----------
try:
    swa_curves = get_metric("SWA")
    cwa_curves = get_metric("CWA")
    hwa_curves = get_metric("HWA")
    if swa_curves and cwa_curves and hwa_curves:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        for lr, curve in swa_curves.items():
            axes[0].plot(range(1, len(curve) + 1), curve, label=f"lr={lr}")
        axes[0].set_title("SWA")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Score")
        axes[0].legend()

        for lr, curve in cwa_curves.items():
            axes[1].plot(range(1, len(curve) + 1), curve, label=f"lr={lr}")
        axes[1].set_title("CWA")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        for lr, curve in hwa_curves.items():
            axes[2].plot(range(1, len(curve) + 1), curve, label=f"lr={lr}")
        axes[2].set_title("HWA")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()

        fig.suptitle(
            "SPR_BENCH Weighted Accuracy Metrics\nLeft: SWA, Middle: CWA, Right: HWA"
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = os.path.join(working_dir, "spr_bench_weighted_accuracy_curves.png")
        plt.savefig(fname)
    else:
        print("Weighted accuracy data unavailable, skipping accuracy plot.")
    plt.close()
except Exception as e:
    print(f"Error creating weighted accuracy curves: {e}")
    plt.close()
