import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

# helper: safely fetch lists
loss_tr = exp.get("losses", {}).get("train", [])
loss_val = exp.get("losses", {}).get("val", [])
metrics_val = exp.get("metrics", {}).get("val", [])
macro_f1 = [m["macro_f1"] for m in metrics_val] if metrics_val else []
cwa_vals = [m["cwa"] for m in metrics_val] if metrics_val else []
epochs = np.arange(1, len(loss_tr) + 1)

# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train Loss", color="tab:blue")
    plt.plot(epochs, loss_val, label="Val Loss", color="tab:orange", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) Macro-F1 curve
try:
    plt.figure()
    plt.plot(epochs, macro_f1, label="Val Macro-F1", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Validation Macro-F1 over Epochs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_macro_f1_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Macro-F1 curve: {e}")
    plt.close()

# 3) CWA curve
try:
    plt.figure()
    plt.plot(epochs, cwa_vals, label="Val CWA", color="tab:red")
    plt.xlabel("Epoch")
    plt.ylabel("Cost-Weighted Accuracy")
    plt.title("SPR_BENCH: Validation CWA over Epochs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_cwa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CWA curve: {e}")
    plt.close()

# 4) Confusion matrix at final epoch
try:
    preds = np.array(exp.get("predictions", []))
    gts = np.array(exp.get("ground_truth", []))
    if preds.size and gts.size:
        labels = sorted(np.unique(np.concatenate([preds, gts])))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH Confusion Matrix (Final Epoch)")
        plt.colorbar()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# Print final metrics
if macro_f1 and cwa_vals:
    print(f"Final Macro-F1: {macro_f1[-1]:.4f}")
    print(f"Final CWA     : {cwa_vals[-1]:.4f}")
