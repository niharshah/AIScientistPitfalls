import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load data -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    wd_dict = experiment_data["weight_decay"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    wd_dict = {}


# helper to make x/y dicts
def collect(key_outer, key_inner):
    return {tag: wd_dict[tag][key_outer][key_inner] for tag in wd_dict}


# ---- Figure 1: training loss ----
try:
    plt.figure()
    for tag, ys in collect("losses", "train").items():
        xs = wd_dict[tag]["epochs"]
        plt.plot(xs, ys, label=tag)
    plt.title(
        "Training Loss over Epochs\nDataset: SPR (synthetic fallback if real unavailable)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_training_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# ---- Figure 2: validation loss ----
try:
    plt.figure()
    for tag, ys in collect("losses", "val").items():
        xs = wd_dict[tag]["epochs"]
        plt.plot(xs, ys, label=tag)
    plt.title("Validation Loss over Epochs\nDataset: SPR")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_validation_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()

# ---- Figure 3: training Cpx-WA ----
try:
    plt.figure()
    for tag, ys in collect("metrics", "train").items():
        xs = wd_dict[tag]["epochs"]
        plt.plot(xs, ys, label=tag)
    plt.title("Training Complexity-Weighted Accuracy\nDataset: SPR")
    plt.xlabel("Epoch")
    plt.ylabel("Cpx-WA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_training_cpxwa_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating training metric plot: {e}")
    plt.close()

# ---- Figure 4: validation Cpx-WA ----
try:
    plt.figure()
    for tag, ys in collect("metrics", "val").items():
        xs = wd_dict[tag]["epochs"]
        plt.plot(xs, ys, label=tag)
    plt.title("Validation Complexity-Weighted Accuracy\nDataset: SPR")
    plt.xlabel("Epoch")
    plt.ylabel("Cpx-WA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_validation_cpxwa_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating validation metric plot: {e}")
    plt.close()

# ---- Figure 5: test Cpx-WA bar chart ----
try:
    plt.figure()
    tags = list(wd_dict.keys())
    scores = [wd_dict[tag]["test_metric"] for tag in tags]
    plt.bar(range(len(tags)), scores, tick_label=tags)
    plt.title("Test Complexity-Weighted Accuracy by Weight Decay\nDataset: SPR")
    plt.ylabel("Cpx-WA")
    fname = os.path.join(working_dir, "SPR_test_cpxwa_bars.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating test metric bar chart: {e}")
    plt.close()
