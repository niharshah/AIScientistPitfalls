import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

data = experiment_data.get("SPR_BENCH")
if data is None:
    print("SPR_BENCH data not found in experiment_data.npy")
    exit()

epochs = np.asarray(data["epochs"])
train_f1 = np.asarray(data["metrics"]["train_f1"])
val_f1 = np.asarray(data["metrics"]["val_f1"])
train_loss = np.asarray(data["losses"]["train"])
val_loss = np.asarray(data["losses"]["val"])
preds = np.asarray(data["predictions"])
gts = np.asarray(data["ground_truth"])
test_f1 = float(data["metrics"]["test_f1"])
sga = float(data["metrics"]["SGA"])

# ---------- Plot 1: F1 curves ----------
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

# ---------- Plot 2: Loss curves ----------
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

# ---------- Plot 3: Test F1 & SGA bar ----------
try:
    plt.figure()
    metrics = ["Test_F1", "SGA"]
    values = [test_f1, sga]
    plt.bar(metrics, values, color=["skyblue", "lightgreen"])
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Test Macro-F1 and SGA")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_testF1_SGA_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# ---------- Plot 4: Confusion matrix ----------
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
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- Print summary ----------
print(f"Test Macro-F1: {test_f1:.4f}")
print(f"Systematic Generalization Accuracy: {sga:.4f}")
