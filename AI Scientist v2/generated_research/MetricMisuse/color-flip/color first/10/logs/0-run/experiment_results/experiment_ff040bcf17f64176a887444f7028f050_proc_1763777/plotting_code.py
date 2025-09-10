import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
exp_path_try = [
    os.path.join(working_dir, "experiment_data.npy"),
    os.path.join(os.getcwd(), "experiment_data.npy"),
]
experiment_data = None
for p in exp_path_try:
    try:
        experiment_data = np.load(p, allow_pickle=True).item()
        break
    except Exception:
        continue
if experiment_data is None:
    raise FileNotFoundError("experiment_data.npy not found in expected locations.")

data = experiment_data["shape_only"]["SPR_BENCH"]
m = data["metrics"]
epochs = np.arange(1, len(m["train_loss"]) + 1)

# ---------- plot 1: loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, m["train_loss"], label="Train Loss")
    plt.plot(epochs, m["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary-Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nTrain vs Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- plot 2: weighted accuracies ----------
try:
    plt.figure()
    plt.plot(epochs, m["val_CWA"], label="CWA")
    plt.plot(epochs, m["val_SWA"], label="SWA")
    plt.plot(epochs, m["val_CWA2"], label="CWA2")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Validation Weighted Accuracies")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_weighted_accuracies.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- plot 3: confusion matrix ----------
try:
    gt = np.array(data["ground_truth"])
    pr = np.array(data["predictions"])
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(gt, pr):
        cm[int(t), int(p)] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- print final metrics ----------
final_idx = -1
print("Final Validation Metrics:")
print(f"  Val Loss: {m['val_loss'][final_idx]:.4f}")
print(f"  CWA     : {m['val_CWA'][final_idx]:.4f}")
print(f"  SWA     : {m['val_SWA'][final_idx]:.4f}")
print(f"  CWA2    : {m['val_CWA2'][final_idx]:.4f}")
