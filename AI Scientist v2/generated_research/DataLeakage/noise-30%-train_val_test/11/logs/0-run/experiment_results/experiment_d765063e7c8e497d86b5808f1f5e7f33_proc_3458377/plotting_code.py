import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_key = "SPR_BENCH"
d = experiment_data.get(ds_key, {})

# 1) Loss curve
try:
    epochs = d.get("epochs", [])
    tr_loss = d.get("losses", {}).get("train", [])
    val_loss = d.get("losses", {}).get("val", [])
    if epochs and tr_loss and val_loss:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.title(f"{ds_key} Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key.lower()}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Macro-F1 curve
try:
    tr_f1 = d.get("metrics", {}).get("train", [])
    val_f1 = d.get("metrics", {}).get("val", [])
    if epochs and tr_f1 and val_f1:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        plt.title(f"{ds_key} Macro-F1 Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_key.lower()}_macro_f1_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# 3) Confusion matrix on test split
try:
    y_true = np.array(d.get("ground_truth", []))
    y_pred = np.array(d.get("predictions", []))
    if y_true.size and y_pred.size and y_true.shape == y_pred.shape:
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title(f"{ds_key} Normalized Confusion Matrix")
        fname = os.path.join(working_dir, f"{ds_key.lower()}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close(fig)
        print(
            f"Final macro-F1 (recomputed): {f1_score(y_true, y_pred, average='macro'):.4f}"
        )
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
