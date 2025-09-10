import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score

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
    print("SPR_BENCH not found in experiment_data.npy")
    exit()

epochs = np.asarray(data["epochs"])
train_loss = np.asarray(data["losses"]["train"])
val_loss = np.asarray(data["losses"]["val"])
train_f1 = np.asarray(data["metrics"]["train_f1"])
val_f1 = np.asarray(data["metrics"]["val_f1"])
test_f1 = data["metrics"].get("test_f1")
SGA = data["metrics"].get("SGA")
preds = np.asarray(data["predictions"])
gts = np.asarray(data["ground_truth"])

# ---------- plot 1: loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- plot 2: F1 curves ----------
try:
    plt.figure()
    plt.plot(epochs, train_f1, label="Train")
    plt.plot(epochs, val_f1, label="Validation", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 Curves\nLeft: Train, Right: Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# ---------- plot 3: bar chart of final F1s ----------
try:
    plt.figure()
    names = ["Train", "Validation", "Test"]
    values = [
        train_f1[-1] if train_f1.size else np.nan,
        val_f1[-1] if val_f1.size else np.nan,
        test_f1 if test_f1 is not None else np.nan,
    ]
    plt.bar(names, values, color="skyblue")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1)
    plt.title("SPR_BENCH Final Macro-F1 Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# ---------- plot 4: confusion matrix ----------
try:
    if preds.size and gts.size:
        cm = confusion_matrix(gts, preds)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("SPR_BENCH Confusion Matrix\nTest Split")
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
                    color=("white" if cm[i, j] > cm.max() / 2 else "black"),
                )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- metric summary ----------
print("==== SPR_BENCH Metric Summary ====")
if train_f1.size:
    print(f"  Final Train Macro-F1:      {train_f1[-1]:.4f}")
if val_f1.size:
    print(f"  Final Validation Macro-F1: {val_f1[-1]:.4f}")
if test_f1 is not None:
    print(f"  Test Macro-F1:             {test_f1:.4f}")
if SGA is not None:
    print(f"  Systematic Gen. Accuracy:  {SGA:.4f}")
