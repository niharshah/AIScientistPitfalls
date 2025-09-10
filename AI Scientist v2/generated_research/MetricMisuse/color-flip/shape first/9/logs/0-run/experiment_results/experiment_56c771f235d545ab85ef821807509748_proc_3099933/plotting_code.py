import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load experiment data ------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ------------ helper to fetch values ----------
def get_vals(wd_key, kind, split):
    """kind: losses|metrics, split: train|val"""
    return experiment_data["weight_decay"][wd_key][kind][split]


# ------------ per-weight-decay plots ----------
for wd_key in experiment_data.get("weight_decay", {}):
    # ----- loss curve -----
    try:
        plt.figure()
        epochs = np.arange(1, len(get_vals(wd_key, "losses", "train")) + 1)
        plt.plot(epochs, get_vals(wd_key, "losses", "train"), label="Train Loss")
        plt.plot(epochs, get_vals(wd_key, "losses", "val"), label="Val Loss")
        plt.title(f"SPR_BENCH Loss Curve (wd={wd_key})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"SPR_BENCH_loss_curve_wd_{wd_key}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for wd={wd_key}: {e}")
        plt.close()

    # ----- metric curve -----
    try:
        plt.figure()
        epochs = np.arange(1, len(get_vals(wd_key, "metrics", "val")) + 1)
        plt.plot(epochs, get_vals(wd_key, "metrics", "val"), marker="o")
        plt.title(f"SPR_BENCH CWA-2D Validation Curve (wd={wd_key})")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        fname = f"SPR_BENCH_CWA_curve_wd_{wd_key}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for wd={wd_key}: {e}")
        plt.close()

# ------------ aggregate comparison ------------
try:
    plt.figure()
    wd_keys, final_scores = [], []
    for wd_key, sub in experiment_data.get("weight_decay", {}).items():
        wd_keys.append(float(wd_key))
        final_scores.append(sub["metrics"]["val"][-1] if sub["metrics"]["val"] else 0.0)
    plt.scatter(wd_keys, final_scores)
    for x, y in zip(wd_keys, final_scores):
        plt.text(x, y, f"{y:.3f}")
    plt.title("SPR_BENCH Final CWA-2D vs Weight Decay")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final Validation CWA-2D")
    plt.xscale("log")
    fname = "SPR_BENCH_final_CWA_vs_weight_decay.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating aggregate plot: {e}")
    plt.close()
