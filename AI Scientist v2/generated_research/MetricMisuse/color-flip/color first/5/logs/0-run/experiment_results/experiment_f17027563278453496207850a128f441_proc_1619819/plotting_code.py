import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds = "SPR_BENCH"
if ds not in experiment_data:
    print(f"Dataset {ds} not found in experiment_data.")
    exit()


# ------------------------------------------------------------------
# Helper to fetch arrays safely
def get_losses(split):
    return np.array(experiment_data[ds]["losses"].get(split, []), dtype=float)


def get_metric_over_epochs(key):
    return np.array(
        [m[key] for m in experiment_data[ds]["metrics"]["val"]], dtype=float
    )


# ------------------------------------------------------------------
# 1) Loss curves ----------------------------------------------------
try:
    train_loss = get_losses("train")
    val_loss = get_losses("val")
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Validation metrics curves -------------------------------------
try:
    acc = get_metric_over_epochs("acc")
    cwa = get_metric_over_epochs("cwa")
    swa = get_metric_over_epochs("swa")
    comp = get_metric_over_epochs("compwa")
    epochs = np.arange(1, len(acc) + 1)

    plt.figure()
    for y, lbl in zip([acc, cwa, swa, comp], ["ACC", "CWA", "SWA", "CompWA"]):
        plt.plot(epochs, y, marker="o", label=lbl)
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("SPR_BENCH Validation Metrics Over Epochs")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_validation_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Confusion matrix (test) ---------------------------------------
try:
    y_true = np.array(experiment_data[ds]["ground_truth"], dtype=int)
    y_pred = np.array(experiment_data[ds]["predictions"], dtype=int)
    n_cls = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("SPR_BENCH Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    tick_labels = list(range(n_cls))
    plt.xticks(tick_labels)
    plt.yticks(tick_labels)
    # annotate cells
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print final test metrics -----------------------------------------
test_metrics = experiment_data[ds]["metrics"].get("test", {})
if test_metrics:
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
