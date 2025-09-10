import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

ed = experiment_data["RemoveClusterFeat"]["SPR_BENCH"]
loss_tr = np.asarray(ed["losses"]["train"])
loss_val = np.asarray(ed["losses"]["val"])
val_metrics = ed["metrics"]["val"]  # list of dicts
epochs = np.arange(1, len(loss_tr) + 1)

val_acc = np.array([m["acc"] for m in val_metrics])
val_cwa = np.array([m["cwa"] for m in val_metrics])
val_swa = np.array([m["swa"] for m in val_metrics])
val_ccwa = np.array([m["ccwa"] for m in val_metrics])

test_metrics = ed["metrics"]["test"]
y_true = np.asarray(ed["ground_truth"])
y_pred = np.asarray(ed["predictions"])
num_labels = len(np.unique(y_true))

# ------------------ plot 1: loss curves ------------------
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train Loss")
    plt.plot(epochs, loss_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Train vs. Val Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------ plot 2: validation accuracy ------------------
try:
    plt.figure()
    plt.plot(epochs, val_acc, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH – Validation Accuracy over Epochs")
    fname = os.path.join(working_dir, "SPR_BENCH_val_acc_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val accuracy plot: {e}")
    plt.close()

# ------------------ plot 3: validation CWA & SWA ------------------
try:
    plt.figure()
    plt.plot(epochs, val_cwa, marker="s", label="CWA")
    plt.plot(epochs, val_swa, marker="^", label="SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("SPR_BENCH – CWA & SWA on Validation Set")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_cwa_swa_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating weighted accuracy plot: {e}")
    plt.close()

# ------------------ plot 4: test metrics bar plot ------------------
try:
    plt.figure()
    names = ["ACC", "CWA", "SWA", "CCWA"]
    values = [
        test_metrics["acc"],
        test_metrics["cwa"],
        test_metrics["swa"],
        test_metrics["ccwa"],
    ]
    plt.bar(names, values, color=["steelblue", "orange", "green", "red"])
    plt.ylim(0, 1)
    plt.title("SPR_BENCH – Test Metrics")
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar plot: {e}")
    plt.close()

# ------------------ plot 5: confusion matrix ------------------
try:
    cm = np.zeros((num_labels, num_labels), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        "SPR_BENCH – Confusion Matrix\nLeft: Ground Truth, Right: Generated Predictions"
    )
    for i in range(num_labels):
        for j in range(num_labels):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ------------------ print final test metrics ------------------
print("TEST RESULTS")
for k, v in test_metrics.items():
    print(f"{k.upper():>5}: {v:.3f}")
