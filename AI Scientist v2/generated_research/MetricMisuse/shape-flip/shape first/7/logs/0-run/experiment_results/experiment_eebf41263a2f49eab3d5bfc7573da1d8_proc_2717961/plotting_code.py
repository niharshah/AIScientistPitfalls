import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- load data -------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Quick exit if data missing
if not experiment_data:
    exit()


# Gather keys and sort by hidden size numeric value
def hid_key(k):  # "hid_64" -> 64
    try:
        return int(k.split("_")[-1])
    except Exception:
        return 0


hid_dict = experiment_data.get("gru_hidden_size", {})
sorted_keys = sorted(hid_dict.keys(), key=hid_key)

# --------- Plot 1: Loss curves ---------
try:
    plt.figure()
    for key in sorted_keys:
        train = hid_dict[key]["losses"]["train"]  # list of (epoch, loss)
        val = hid_dict[key]["losses"]["val"]
        epochs_tr = [e for e, _ in train]
        loss_tr = [l for _, l in train]
        loss_val = [l for _, l in val]
        plt.plot(epochs_tr, loss_tr, label=f"{key}-train")
        plt.plot(epochs_tr, loss_val, "--", label=f"{key}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Synthetic SPR: Training vs Validation Loss\nGRU Hidden Size Sweep")
    plt.legend()
    fname = os.path.join(working_dir, "spr_loss_curves_hidden_size.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# --------- Plot 2: HWA curves ---------
try:
    plt.figure()
    for key in sorted_keys:
        train = hid_dict[key]["metrics"]["train"]  # list of (epoch, hwa)
        val = hid_dict[key]["metrics"]["val"]
        epochs_tr = [e for e, _ in train]
        hwa_tr = [h for _, h in train]
        hwa_val = [h for _, h in val]
        plt.plot(epochs_tr, hwa_tr, label=f"{key}-train")
        plt.plot(epochs_tr, hwa_val, "--", label=f"{key}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("Synthetic SPR: Training vs Validation HWA\nGRU Hidden Size Sweep")
    plt.legend()
    fname = os.path.join(working_dir, "spr_hwa_curves_hidden_size.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA plot: {e}")
    plt.close()

# --------- Plot 3: Test HWA bar chart ---------
try:
    plt.figure()
    test_scores = [hid_dict[k].get("test_hwa", 0) for k in sorted_keys]
    xs = np.arange(len(sorted_keys))
    plt.bar(xs, test_scores, color="orange")
    plt.xticks(xs, sorted_keys, rotation=45)
    plt.ylabel("Test Harmonic Weighted Accuracy")
    plt.title("Synthetic SPR: Test HWA by GRU Hidden Size")
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_test_hwa_bar_hidden_size.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test HWA bar plot: {e}")
    plt.close()
