import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up working dir ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ---------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

run_key = "spr_rgcn"
run_data = experiment_data.get(run_key, {})
epochs = run_data.get("epochs", [])
train_loss = run_data.get("losses", {}).get("train", [])
val_loss = run_data.get("losses", {}).get("val", [])
val_compwa = run_data.get("metrics", {}).get("val_compwa", [])
test_compwa = run_data.get("metrics", {}).get("test_compwa", None)
dataset_type = experiment_data.get("dataset_type", "SPR_synth")

# ---------- plotting ----------------------------------------------------------
# 1. Loss curve
try:
    if epochs and train_loss and val_loss:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.title(f"Loss Curve (Dataset: {dataset_type}, Model: {run_key})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"loss_curve_{dataset_type}_{run_key}.png"
        plt.savefig(os.path.join(working_dir, fname))
except Exception as e:
    print(f"Error creating loss curve: {e}")
finally:
    plt.close()

# 2. Validation CompWA curve
try:
    if epochs and val_compwa:
        plt.figure()
        plt.plot(epochs, val_compwa, marker="o")
        plt.title(f"Validation CompWA (Dataset: {dataset_type}, Model: {run_key})")
        plt.xlabel("Epoch")
        plt.ylabel("CompWA")
        fname = f"val_compwa_curve_{dataset_type}_{run_key}.png"
        plt.savefig(os.path.join(working_dir, fname))
except Exception as e:
    print(f"Error creating CompWA curve: {e}")
finally:
    plt.close()

# 3. Test CompWA summary bar
try:
    if test_compwa is not None:
        plt.figure()
        plt.bar(["Test"], [test_compwa])
        plt.title(f"Test CompWA (Dataset: {dataset_type}, Model: {run_key})")
        plt.ylabel("CompWA")
        fname = f"test_compwa_{dataset_type}_{run_key}.png"
        plt.savefig(os.path.join(working_dir, fname))
except Exception as e:
    print(f"Error creating test CompWA bar: {e}")
finally:
    plt.close()

# ---------- print evaluation metric ------------------------------------------
if test_compwa is not None:
    print(f"Test Complexity-Weighted Accuracy: {test_compwa:.4f}")
