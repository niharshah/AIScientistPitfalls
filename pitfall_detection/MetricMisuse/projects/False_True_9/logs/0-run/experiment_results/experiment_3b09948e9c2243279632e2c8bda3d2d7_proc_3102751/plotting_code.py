import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- setup & load -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

rec = experiment_data.get("SPR_BENCH", {})
loss_train = rec.get("losses", {}).get("train", [])
loss_val = rec.get("losses", {}).get("val", [])
metrics = rec.get("metrics", {}).get("val", [])
preds = np.array(rec.get("predictions", []))
gts = np.array(rec.get("ground_truth", []))

if not (loss_train and loss_val and metrics and preds.size):
    print("Incomplete SPR_BENCH data – nothing to plot.")
    exit()

epochs_loss_t, vals_loss_t = zip(*loss_train)
epochs_loss_v, vals_loss_v = zip(*loss_val)
ep_m, swa, cwa, dawa = zip(*metrics)
num_classes = int(max(np.max(preds), np.max(gts)) + 1)

# ---------------- plot 1: loss curves ----------
try:
    plt.figure()
    plt.plot(epochs_loss_t, vals_loss_t, label="Train")
    plt.plot(epochs_loss_v, vals_loss_v, "--", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------------- plot 2: metric curves --------
try:
    plt.figure()
    plt.plot(ep_m, swa, label="SWA")
    plt.plot(ep_m, cwa, label="CWA")
    plt.plot(ep_m, dawa, label="DAWA")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Weighted Accuracies Over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_metric_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric curve: {e}")
    plt.close()

# ---------------- plot 3: final bar chart ------
try:
    plt.figure()
    plt.bar(["SWA", "CWA"], [swa[-1], cwa[-1]], color=["steelblue", "salmon"])
    plt.ylim(0, 1)
    plt.ylabel("Final-Epoch Accuracy")
    plt.title(f"SPR_BENCH: Final Accuracies (DAWA={dawa[-1]:.3f})")
    fname = os.path.join(working_dir, "SPR_BENCH_final_accuracy_bars.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy bar chart: {e}")
    plt.close()

# ---------------- plot 4: confusion matrix -----
try:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Bottom: Predicted")
    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes))
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------- plot 5: class distribution ---
try:
    plt.figure()
    bins = np.arange(num_classes + 1) - 0.5
    plt.hist(gts, bins=bins, alpha=0.6, label="Ground Truth")
    plt.hist(preds, bins=bins, alpha=0.6, label="Predictions")
    plt.xlabel("Class Index")
    plt.ylabel("Count")
    plt.title("SPR_BENCH: Class Distribution Comparison")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_class_distribution.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating class distribution histogram: {e}")
    plt.close()

# ---------------- print final numbers ----------
print(
    f"Final Epoch Metrics — SWA: {swa[-1]:.4f}, CWA: {cwa[-1]:.4f}, DAWA: {dawa[-1]:.4f}"
)
