import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def safe_load():
    try:
        path = os.path.join(working_dir, "experiment_data.npy")
        return np.load(path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading experiment data: {e}")
        return None


exp = safe_load()
if exp is None:
    exit()

run = exp["no_seq_edges"]["SPR"]
epochs = run["epochs"]
tr_loss = run["losses"]["train"]
val_loss = run["losses"]["val"]
val_mets = run["metrics"]["val"]  # list of dicts per epoch
cwa = [m["CWA"] for m in val_mets]
swa = [m["SWA"] for m in val_mets]
hpa = [m["HPA"] for m in val_mets]
preds = np.array(run["predictions"])
gts = np.array(run["ground_truth"])
num_classes = len(np.unique(gts))

# -------- plot 1: loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, tr_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Dataset – Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------- plot 2: metric curves ----------
try:
    plt.figure()
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, hpa, label="HPA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR Dataset – Validation Metrics per Epoch")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_metric_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# -------- plot 3: confusion matrix ----------
try:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SPR Dataset – Confusion Matrix (Test Set)")
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=8,
            )
    fname = os.path.join(working_dir, "SPR_confusion_matrix.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------- evaluation metric printout ----------
test_acc = (preds == gts).mean()
print(f"Test Accuracy: {test_acc:.3f}")
