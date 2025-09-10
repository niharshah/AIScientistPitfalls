import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- Plot 1: Validation HWA vs Epoch ----------
try:
    plt.figure()
    hidden_dict = experiment_data.get("hidden_size", {})
    for hid_str, info in hidden_dict.items():
        epochs, hwa_val = zip(*info["metrics"]["val"])
        plt.plot(epochs, hwa_val, label=f"hid={hid_str}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation HWA")
    plt.title("Validation Harmonic-Weighted-Accuracy vs Epoch (Synthetic SPR)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_SYNTH_val_HWA_vs_epoch.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Validation HWA plot: {e}")
    plt.close()

# ---------- Plot 2: Loss Curves (Train & Val) ----------
try:
    plt.figure()
    for hid_str, info in hidden_dict.items():
        ep_tr, loss_tr = zip(*info["losses"]["train"])
        ep_val, loss_val = zip(*info["losses"]["val"])
        plt.plot(ep_tr, loss_tr, linestyle="--", label=f"Train hid={hid_str}")
        plt.plot(ep_val, loss_val, linestyle="-", label=f"Val hid={hid_str}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training & Validation Loss (Synthetic SPR)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_SYNTH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Loss Curves plot: {e}")
    plt.close()

# ---------- Plot 3: Test HWA vs Hidden Size ----------
try:
    plt.figure()
    hids, test_hwa = [], []
    for hid_str, info in hidden_dict.items():
        hids.append(int(hid_str))
        test_hwa.append(info["test_hwa"])
    order = np.argsort(hids)
    hids_sorted = np.array(hids)[order]
    hwa_sorted = np.array(test_hwa)[order]
    plt.bar([str(h) for h in hids_sorted], hwa_sorted)
    plt.xlabel("Hidden Size")
    plt.ylabel("Test HWA")
    plt.title("Test Harmonic-Weighted-Accuracy across Hidden Sizes (Synthetic SPR)")
    fname = os.path.join(working_dir, "SPR_SYNTH_test_HWA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Test HWA bar plot: {e}")
    plt.close()
