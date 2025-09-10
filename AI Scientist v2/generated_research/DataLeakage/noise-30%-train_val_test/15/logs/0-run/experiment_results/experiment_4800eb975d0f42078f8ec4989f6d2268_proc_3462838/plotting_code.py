import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to get ordered keys
def _ordered_keys(d):
    return sorted(d.keys(), key=lambda k: int(k.split("_")[-1]))


# -------------------------------------------------------------------------
# Figure 1: Loss curves (train / val)
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key in _ordered_keys(experiment_data):
        rec = experiment_data[key]["SPR_BENCH"]
        axes[0].plot(rec["epochs"], rec["losses"]["train"], label=key)
        axes[1].plot(rec["epochs"], rec["losses"]["val"], label=key)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation")
    fig.suptitle("SPR_BENCH Loss Curves (Left: Train, Right: Validation)")
    fig.legend(loc="upper center", ncol=4)
    out_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# -------------------------------------------------------------------------
# Figure 2: Macro-F1 curves (train / val)
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key in _ordered_keys(experiment_data):
        rec = experiment_data[key]["SPR_BENCH"]
        axes[0].plot(rec["epochs"], rec["metrics"]["train"], label=key)
        axes[1].plot(rec["epochs"], rec["metrics"]["val"], label=key)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_title("Train")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Validation")
    fig.suptitle("SPR_BENCH Macro-F1 Curves (Left: Train, Right: Validation)")
    fig.legend(loc="upper center", ncol=4)
    out_path = os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves plot: {e}")
    plt.close()

# -------------------------------------------------------------------------
# Figure 3: Test Macro-F1 bar chart
try:
    keys = _ordered_keys(experiment_data)
    dims = [int(k.split("_")[-1]) for k in keys]
    test_f1 = [experiment_data[k]["SPR_BENCH"]["test_macroF1"] for k in keys]
    plt.figure(figsize=(6, 4))
    plt.bar([str(d) for d in dims], test_f1, color="skyblue")
    plt.xlabel("d_model")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Test Macro-F1 by d_model")
    out_path = os.path.join(working_dir, "SPR_BENCH_test_macroF1_bar.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()
except Exception as e:
    print(f"Error creating test F1 bar plot: {e}")
    plt.close()
