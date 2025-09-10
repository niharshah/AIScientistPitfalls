import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("Remove_BoC_Branch", {}).get("SPR_BENCH", {})

loss_tr = ed.get("losses", {}).get("train", [])
loss_val = ed.get("losses", {}).get("val", [])
f1_tr = ed.get("metrics", {}).get("train_f1", [])
f1_val = ed.get("metrics", {}).get("val_f1", [])
preds = ed.get("predictions", [])
gts = ed.get("ground_truth", [])

# ---------------- plot 1: loss curve ----------------
try:
    plt.figure()
    epochs = range(1, len(loss_tr) + 1)
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------------- plot 2: F1 curve ----------------
try:
    plt.figure()
    plt.plot(epochs, f1_tr, label="Train")
    plt.plot(epochs, f1_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Training vs Validation Macro-F1")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# ---------------- plot 3: confusion matrix ----------------
try:
    if preds and gts:
        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix (Test Split)")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------- metrics ----------------
if preds and gts:
    test_f1 = f1_score(gts, preds, average="macro")
    print(f"Final Test Macro-F1: {test_f1:.4f}")
print(f"Rule-Extraction Dev Acc: {ed.get('metrics', {}).get('REA_dev', 'N/A')}")
print(f"Rule-Extraction Test Acc: {ed.get('metrics', {}).get('REA_test', 'N/A')}")
