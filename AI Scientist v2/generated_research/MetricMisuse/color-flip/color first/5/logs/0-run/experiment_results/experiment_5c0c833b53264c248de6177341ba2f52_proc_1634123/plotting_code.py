import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["epochs_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

epochs = np.arange(1, len(ed["losses"]["train"]) + 1)
val_metrics = ed["metrics"]["val"]  # list of dicts

# ---------- Plot 1: loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
    plt.plot(epochs, ed["losses"]["val"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train Loss, Right: Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- Plot 2: validation accuracies ----------
try:
    plt.figure()
    acc = [m["acc"] for m in val_metrics]
    cwa = [m["cwa"] for m in val_metrics]
    swa = [m["swa"] for m in val_metrics]
    compwa = [m["compwa"] for m in val_metrics]
    plt.plot(epochs, acc, label="ACC")
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, compwa, label="CompWA")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("SPR_BENCH Validation Metrics\nMultiple weighted accuracies over epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()

# ---------- Plot 3: confusion matrix ----------
try:
    preds = np.array(ed["predictions"])
    gts = np.array(ed["ground_truth"])
    num_classes = int(max(preds.max(), gts.max()) + 1)
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(gts, preds):
        conf[t, p] += 1
    plt.figure()
    im = plt.imshow(conf, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- Plot 4: test metrics bar chart ----------
try:
    test_metrics = ed["metrics"]["test"]
    names = list(test_metrics.keys())
    values = [test_metrics[k] for k in names]
    plt.figure()
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.title("SPR_BENCH Test Metrics\nBar chart of final evaluation scores")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar chart: {e}")
    plt.close()

# ---------- print evaluation metrics ----------
print("Final Test Metrics:")
for k, v in ed["metrics"]["test"].items():
    print(f"{k}: {v:.3f}")
