import matplotlib.pyplot as plt
import numpy as np
import os

# --------- paths & loading ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data = experiment_data.get("SPR_BENCH", None)
if data is None:
    print("No SPR_BENCH data found.")
    exit()

epochs = np.array(data["epochs"])
pre_settings = np.array(data["pretraining_setting"])
train_loss = np.array(data["losses"]["train"])
val_loss = np.array(data["losses"]["val"])
val_mwa = np.array(data["metrics"]["val_MWA"])

# -------- figure 1: loss curves --------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------- figure 2: validation MWA curve --------
try:
    plt.figure()
    plt.plot(epochs, val_mwa, marker="o")
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("Validation MWA")
    plt.title("SPR_BENCH Validation MWA Across Epochs")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_MWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating MWA curve plot: {e}")
    plt.close()

# -------- figure 3: final MWA vs pre-training setting --------
try:
    # take final epoch for each distinct pretraining setting
    summary = {}
    for ep, pre, mwa in zip(epochs, pre_settings, val_mwa):
        summary[pre] = mwa  # later epochs overwrite earlier ones, leaving final
    pres = np.array(list(summary.keys()))
    mwas = np.array(list(summary.values()))
    plt.figure()
    plt.bar(pres.astype(str), mwas)
    plt.xlabel("Pre-training Epochs")
    plt.ylabel("Final Validation MWA")
    plt.title("SPR_BENCH Final MWA vs Pre-training Setting")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_MWA_vs_pretraining.png"))
    plt.close()
except Exception as e:
    print(f"Error creating summary plot: {e}")
    plt.close()

# -------- print overall best metric --------
best_mwa = np.nanmax(val_mwa)
print(f"Best Validation MWA observed: {best_mwa:.4f}")
