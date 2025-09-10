import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds = "SPR_BENCH"
if ds not in experiment_data:
    print(f"Dataset {ds} not found in experiment_data, exiting.")
    exit()

epochs = experiment_data[ds]["epochs"]
loss_train = experiment_data[ds]["losses"]["train"]
loss_val = experiment_data[ds]["losses"]["val"]
swa = experiment_data[ds]["metrics"]["SWA"]
cwa = experiment_data[ds]["metrics"]["CWA"]
scwa = experiment_data[ds]["metrics"]["SCWA"]
preds = experiment_data[ds]["predictions"]
gts = experiment_data[ds]["ground_truth"]

# ---------- Plot Loss Curves ----------
try:
    plt.figure()
    plt.plot(epochs, loss_train, label="Train Loss")
    plt.plot(epochs, loss_val, label="Val Loss")
    plt.title("Loss vs Epochs (SPR_BENCH)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- Plot Metric Curves ----------
try:
    plt.figure()
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, scwa, label="SCWA")
    plt.title("Weighted Accuracies vs Epochs (SPR_BENCH)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_metric_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric curves: {e}")
    plt.close()

# ---------- Confusion Matrix ----------
try:
    cm = np.zeros((2, 2), dtype=int)
    for gt, pr in zip(gts, preds):
        cm[gt, pr] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.title("Confusion Matrix (SPR_BENCH)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- Print final metrics ----------
if epochs:
    print(f"Final epoch ({epochs[-1]}) metrics:")
    print(f"  SWA  = {swa[-1]:.4f}")
    print(f"  CWA  = {cwa[-1]:.4f}")
    print(f"  SCWA = {scwa[-1]:.4f}")
