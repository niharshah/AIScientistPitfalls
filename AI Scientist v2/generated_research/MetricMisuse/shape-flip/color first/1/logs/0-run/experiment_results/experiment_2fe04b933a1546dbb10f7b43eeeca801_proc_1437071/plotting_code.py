import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------
# Load experiment data
# -------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
if dataset not in experiment_data:
    print(f"{dataset} not found in experiment data.")
    exit()

data = experiment_data[dataset]
epochs = range(1, len(data["losses"]["train"]) + 1)

# -------------------------------------------------------------
# 1) Loss curves
# -------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, data["losses"]["train"], label="Train")
    plt.plot(epochs, data["losses"]["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# -------------------------------------------------------------
# 2) Accuracy curves
# -------------------------------------------------------------
try:
    tr_acc = [m["acc"] for m in data["metrics"]["train"]]
    val_acc = [m["acc"] for m in data["metrics"]["val"]]
    plt.figure()
    plt.plot(epochs, tr_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# -------------------------------------------------------------
# 3) CAA curves
# -------------------------------------------------------------
try:
    tr_caa = [m["caa"] for m in data["metrics"]["train"]]
    val_caa = [m["caa"] for m in data["metrics"]["val"]]
    plt.figure()
    plt.plot(epochs, tr_caa, label="Train")
    plt.plot(epochs, val_caa, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("CAA")
    plt.title("SPR_BENCH Complexity-Adjusted Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_CAA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CAA curve: {e}")
    plt.close()

# -------------------------------------------------------------
# 4) Final test metric bar chart
# -------------------------------------------------------------
try:
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])
    test_acc = (preds == gts).mean()
    plt.figure()
    plt.bar(["Accuracy"], [test_acc])
    plt.ylim(0, 1)
    plt.title("SPR_BENCH Test Set Accuracy")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar chart: {e}")
    plt.close()

# -------------------------------------------------------------
# 5) Confusion matrix
# -------------------------------------------------------------
try:
    labels = sorted(set(gts))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title("SPR_BENCH Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------------------------------------------------------------
# Print final metric
# -------------------------------------------------------------
print(f"SPR_BENCH Test Accuracy: {test_acc:.3f}")
