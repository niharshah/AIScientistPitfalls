import matplotlib.pyplot as plt
import numpy as np
import os

# Basic setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data, ed = {}, {}
else:
    ed = experiment_data.get("NoBiDir", {}).get("SPR", {})

epochs = ed.get("epochs", [])
tr_loss = ed.get("losses", {}).get("train", [])
val_loss = ed.get("losses", {}).get("val", [])
tr_met = ed.get("metrics", {}).get("train", [])
val_met = ed.get("metrics", {}).get("val", [])
pred = np.array(ed.get("predictions", []))
gt = np.array(ed.get("ground_truth", []))

# ---------- Loss curve ----------
try:
    plt.figure()
    plt.plot(epochs, tr_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.title("Loss vs Epochs - NoBiDir SPR")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropyLoss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "NoBiDir_SPR_loss_curves.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()


# ---------- Metric curves ----------
def metric_over_epochs(metric_name):
    tr = [m.get(metric_name, np.nan) for m in tr_met]
    va = [m.get(metric_name, np.nan) for m in val_met]
    return tr, va


for metric in ["CWA", "SWA", "HWA"]:
    try:
        tr_vals, val_vals = metric_over_epochs(metric)
        plt.figure()
        plt.plot(epochs, tr_vals, label="Train")
        plt.plot(epochs, val_vals, label="Validation")
        plt.title(f"{metric} vs Epochs - NoBiDir SPR")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        fname = f"NoBiDir_SPR_{metric}_curves.png"
        plt.savefig(os.path.join(working_dir, fname), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating {metric} plot: {e}")
        plt.close()

# ---------- Confusion matrix ----------
try:
    labels = np.sort(np.unique(np.concatenate([gt, pred])))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(gt, pred):
        cm[np.where(labels == t)[0][0], np.where(labels == p)[0][0]] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - NoBiDir SPR (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.savefig(os.path.join(working_dir, "NoBiDir_SPR_confusion_matrix.png"), dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- Print evaluation metric ----------
if gt.size and pred.size:
    accuracy = (gt == pred).mean()
    print(f"Test accuracy: {accuracy:.3f}")
