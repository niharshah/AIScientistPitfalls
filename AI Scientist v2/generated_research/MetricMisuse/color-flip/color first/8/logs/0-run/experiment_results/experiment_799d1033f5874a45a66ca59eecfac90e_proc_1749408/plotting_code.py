import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data["RandomCluster"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = None

# -------------------------- Figure 1 ------------------------
try:
    if run is None:
        raise ValueError("No run data")

    tr_epochs, tr_loss = zip(*run["losses"]["train"])
    va_epochs, va_loss = zip(*run["losses"]["val"])

    plt.figure()
    plt.plot(tr_epochs, tr_loss, label="Train")
    plt.plot(va_epochs, va_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR RandomCluster – Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "RandomCluster_SPR_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# -------------------------- Figure 2 ------------------------
try:
    if run is None:
        raise ValueError("No run data")

    met = np.array(run["metrics"]["val"])  # cols: epoch, CWA, SWA, HM, OCGA
    epochs, cwa, swa, hm, ocga = met.T

    plt.figure()
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, hm, label="Harmonic Mean")
    plt.plot(epochs, ocga, label="OCGA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR RandomCluster – Validation Metrics over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "RandomCluster_SPR_metric_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating metric curves: {e}")
    plt.close()

# -------------------------- Figure 3 ------------------------
try:
    if run is None:
        raise ValueError("No run data")

    preds = np.array(run["predictions"])
    gts = np.array(run["ground_truth"])
    classes = np.unique(np.concatenate([gts, preds]))
    cm = np.zeros((classes.size, classes.size), dtype=int)
    for t, p in zip(gts, preds):
        cm[np.where(classes == t)[0][0], np.where(classes == p)[0][0]] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("SPR RandomCluster – Test Confusion Matrix")
    fname = os.path.join(working_dir, "RandomCluster_SPR_confusion_matrix.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
