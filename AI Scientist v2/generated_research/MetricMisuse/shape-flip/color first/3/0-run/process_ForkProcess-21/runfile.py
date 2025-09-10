import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

loss_tr = np.array(ed["losses"]["train"])
loss_va = np.array(ed["losses"]["val"])
epochs = np.arange(1, len(loss_tr) + 1)


def _metric_arr(key, split):
    return np.array([m[key] for m in ed["metrics"][split]])


bwa_tr, bwa_va = _metric_arr("BWA", "train"), _metric_arr("BWA", "val")
cwa_va, swa_va, scwa_va = (
    _metric_arr("CWA", "val"),
    _metric_arr("SWA", "val"),
    _metric_arr("SCWA", "val"),
)

preds = np.array(ed["predictions"])
gtruth = np.array(ed["ground_truth"])
test_metrics = ed["test_metrics"]

# ---------------- plots ----------------
# 1. Loss curve
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_va, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("spr_bench Loss Curve\nTrain vs Validation")
    plt.legend()
    fpath = os.path.join(working_dir, "spr_bench_loss_curve.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2. BWA curve
try:
    plt.figure()
    plt.plot(epochs, bwa_tr, label="Train")
    plt.plot(epochs, bwa_va, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Weighted Accuracy")
    plt.title("spr_bench BWA Curve\nTrain vs Validation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_BWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating BWA curve: {e}")
    plt.close()

# 3. Validation metric bundle
try:
    plt.figure()
    plt.plot(epochs, bwa_va, label="BWA")
    plt.plot(epochs, cwa_va, label="CWA")
    plt.plot(epochs, swa_va, label="SWA")
    plt.plot(epochs, scwa_va, label="SCWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("spr_bench Validation Metrics")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_val_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()

# 4. Confusion matrix heat-map
try:
    import itertools

    labels = np.unique(np.concatenate([gtruth, preds]))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for gt, pr in zip(gtruth, preds):
        cm[gt, pr] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("spr_bench Confusion Matrix\nCounts of Test Predictions")
    # annotate
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=6)
    plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# 5. Test metric bar chart
try:
    plt.figure()
    names = ["loss", "BWA", "CWA", "SWA", "SCWA"]
    values = [test_metrics[k] for k in names]
    plt.bar(names, values)
    plt.ylabel("Value")
    plt.title("spr_bench Final Test Metrics")
    plt.savefig(os.path.join(working_dir, "spr_bench_test_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar chart: {e}")
    plt.close()

# --------------- print metrics ---------------
print("Final Test Metrics:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}")
