import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------- load data --------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

mdl = "unidirectional_gru"
dset = "spr_bench"
exp = experiment_data.get(mdl, {}).get(dset, {})

# ------------------------------ helper fetch -------------------------------
losses = exp.get("losses", {})
metrics = exp.get("metrics", {})
preds = exp.get("predictions", [])
gts = exp.get("ground_truth", [])

# --------------------------------- plots ----------------------------------
# 1) Loss curves
try:
    plt.figure()
    if losses:
        epochs = range(1, len(losses["train"]) + 1)
        plt.plot(epochs, losses["train"], label="Train Loss")
        plt.plot(epochs, losses["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("spr_bench: Train vs Val Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) HWA curves
try:
    plt.figure()
    if metrics:
        hwa_tr = [m[2] for m in metrics["train"]]
        hwa_val = [m[2] for m in metrics["val"]]
        epochs = range(1, len(hwa_tr) + 1)
        plt.plot(epochs, hwa_tr, label="Train HWA")
        plt.plot(epochs, hwa_val, label="Val HWA")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("spr_bench: Train vs Val Harmonic Weighted Accuracy")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_hwa_curve.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve: {e}")
    plt.close()

# 3) Confusion matrix on test set
try:
    if preds and gts:
        labels = sorted(set(gts))
        cm = confusion_matrix(gts, preds, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("spr_bench: Test Confusion Matrix\nLeft: Ground Truth, Right: Preds")
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# --------------------------- print final metrics ---------------------------
test_met = exp.get("metrics", {}).get("test", None)
if test_met:
    print(f"Test SWA={test_met[0]:.4f}  CWA={test_met[1]:.4f}  HWA={test_met[2]:.4f}")
