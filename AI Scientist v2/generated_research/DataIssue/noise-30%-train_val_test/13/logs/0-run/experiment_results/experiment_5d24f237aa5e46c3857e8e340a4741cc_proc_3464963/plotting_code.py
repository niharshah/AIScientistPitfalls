import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- data load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_dict = experiment_data.get("batch_size_tuning", {}).get("SPR_BENCH", {})
# keep only successful runs
bs_keys = [k for k, v in spr_dict.items() if "error" not in v]
bs_keys_sorted = sorted(bs_keys, key=lambda x: int(x))


# helper: get lists for plotting
def get_list(bs, field, split):
    return spr_dict[bs][field][split]


# ---------- Plot 1: Loss curves ----------
try:
    plt.figure(figsize=(10, 4))
    epochs = spr_dict[bs_keys_sorted[0]]["epochs"]
    # Left subplot: train loss
    ax1 = plt.subplot(1, 2, 1)
    for bs in bs_keys_sorted:
        ax1.plot(epochs, get_list(bs, "losses", "train"), label=f"bs={bs}")
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    # Right subplot: val loss
    ax2 = plt.subplot(1, 2, 2)
    for bs in bs_keys_sorted:
        ax2.plot(epochs, get_list(bs, "losses", "val"), label=f"bs={bs}")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.suptitle("SPR_BENCH Loss Curves\nLeft: Train Loss, Right: Val Loss")
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# ---------- Plot 2: F1 curves ----------
try:
    plt.figure(figsize=(10, 4))
    epochs = spr_dict[bs_keys_sorted[0]]["epochs"]
    # Left subplot: train F1
    ax1 = plt.subplot(1, 2, 1)
    for bs in bs_keys_sorted:
        ax1.plot(epochs, get_list(bs, "metrics", "train_f1"), label=f"bs={bs}")
    ax1.set_title("Train Macro-F1")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("F1")
    ax1.legend()
    # Right subplot: val F1
    ax2 = plt.subplot(1, 2, 2)
    for bs in bs_keys_sorted:
        ax2.plot(epochs, get_list(bs, "metrics", "val_f1"), label=f"bs={bs}")
    ax2.set_title("Validation Macro-F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")
    ax2.legend()
    plt.suptitle("SPR_BENCH Macro-F1 Curves\nLeft: Train F1, Right: Val F1")
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating F1 curves plot: {e}")
    plt.close()

# ---------- Plot 3: Test F1 bar ----------
try:
    test_f1s = [spr_dict[bs]["test_f1"] for bs in bs_keys_sorted]
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(bs_keys_sorted)), test_f1s, tick_label=bs_keys_sorted)
    plt.title("SPR_BENCH Test Macro-F1 vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Macro-F1")
    fname = os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating Test F1 bar plot: {e}")
    plt.close()

# ---------- Print evaluation summary ----------
try:
    for bs, f1 in zip(bs_keys_sorted, test_f1s):
        print(f"Batch size {bs}: Test macro-F1 = {f1:.4f}")
    best_idx = int(np.argmax(test_f1s))
    print(
        f"Best batch size: {bs_keys_sorted[best_idx]} with macro-F1={test_f1s[best_idx]:.4f}"
    )
except Exception as e:
    print(f"Error printing evaluation summary: {e}")
