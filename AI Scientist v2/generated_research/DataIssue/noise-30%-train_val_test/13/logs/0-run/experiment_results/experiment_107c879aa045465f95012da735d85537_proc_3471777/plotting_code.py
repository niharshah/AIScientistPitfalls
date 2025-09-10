import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score

# ---------- load ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

if "SPR_BENCH" not in experiment_data:
    print("SPR_BENCH data not found.")
    raise SystemExit

run = experiment_data["SPR_BENCH"]  # single run dictionary

epochs = np.asarray(run["epochs"])
train_f1 = np.asarray(run["metrics"]["train_f1"])
val_f1 = np.asarray(run["metrics"]["val_f1"])
train_loss = np.asarray(run["losses"]["train"])
val_loss = np.asarray(run["losses"]["val"])
test_f1 = run["test_f1"]
SGA = run["metrics"]["SGA"][0] if run["metrics"]["SGA"] else None
preds = np.asarray(run["predictions"])
gts = np.asarray(run["ground_truth"])

# ---------- plot 1: F1 curves ----------
try:
    plt.figure()
    plt.plot(epochs, train_f1, label="Train F1")
    plt.plot(epochs, val_f1, linestyle="--", label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Training vs Validation Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# ---------- plot 2: Loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, linestyle="--", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- plot 3: Test metrics bar ----------
try:
    plt.figure()
    names = ["Macro-F1", "SGA"] if SGA is not None else ["Macro-F1"]
    vals = [test_f1, SGA] if SGA is not None else [test_f1]
    plt.bar(names, vals, color=["steelblue", "salmon"][: len(vals)])
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Test Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar: {e}")
    plt.close()

# ---------- plot 4: confusion matrix ----------
try:
    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title("SPR_BENCH Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

print(f"Final Test Macro-F1: {test_f1:.4f}")
if SGA is not None:
    print(f"Final SGA: {SGA:.4f}")
