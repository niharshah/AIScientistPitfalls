import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import itertools

# mandatory working dir variable
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    (exp,) = [{}]  # empty fallback to avoid NameError later

# we assume one config / one dataset as in the provided script
cfg_name = next(iter(exp)) if exp else None
dset_name = next(iter(exp[cfg_name])) if cfg_name else None
ed = exp.get(cfg_name, {}).get(dset_name, {}) if cfg_name else {}

epochs = ed.get("epochs", [])
loss_train = ed.get("losses", {}).get("train", [])
pcwa_train = (
    [m["pcwa"] for m in ed.get("metrics", {}).get("train", [])] if epochs else []
)
pcwa_val = [m["pcwa"] for m in ed.get("metrics", {}).get("val", [])] if epochs else []
cwa_val = [m["cwa"] for m in ed.get("metrics", {}).get("val", [])] if epochs else []
swa_val = [m["swa"] for m in ed.get("metrics", {}).get("val", [])] if epochs else []
y_true = ed.get("ground_truth", [])
y_pred = ed.get("predictions", [])

# ------------------------------------------------------------------
# 1. Train vs. Val PCWA
try:
    plt.figure()
    plt.plot(epochs, pcwa_train, label="Train PCWA")
    plt.plot(epochs, pcwa_val, label="Val PCWA")
    plt.title(f"{dset_name} – PCWA over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("PCWA")
    plt.legend()
    fout = os.path.join(working_dir, f"{dset_name}_train_val_PCWA.png")
    plt.savefig(fout)
    plt.close()
except Exception as e:
    print(f"Error creating PCWA plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2. Training Loss
try:
    plt.figure()
    plt.plot(epochs, loss_train, color="tab:orange")
    plt.title(f"{dset_name} – Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    fout = os.path.join(working_dir, f"{dset_name}_train_loss.png")
    plt.savefig(fout)
    plt.close()
except Exception as e:
    print(f"Error creating Loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3. Validation CWA / SWA
try:
    plt.figure()
    plt.plot(epochs, cwa_val, label="Val CWA")
    plt.plot(epochs, swa_val, label="Val SWA")
    plt.title(f"{dset_name} – CWA & SWA over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fout = os.path.join(working_dir, f"{dset_name}_val_CWA_SWA.png")
    plt.savefig(fout)
    plt.close()
except Exception as e:
    print(f"Error creating CWA/SWA plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 4. Confusion Matrix (best epoch)
try:
    if y_true and y_pred:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title(
            f"{dset_name} – Confusion Matrix\n(Left axis: True Labels, Bottom axis: Predicted)"
        )
        plt.colorbar()
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)
        # annotate cells
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
            )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        fout = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
        plt.savefig(fout)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
