import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

ds_key = None
if experiment_data:
    ds_key = next(iter(experiment_data.keys()))  # 'SPR' expected

# -------- Plot 1: loss curves -----------------------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    plt.figure()
    tr = experiment_data[ds_key]["losses"]["train"]
    vl = experiment_data[ds_key]["losses"]["val"]
    if tr:
        e, l = zip(*tr)
        plt.plot(e, l, "--", label="train")
    if vl:
        e, l = zip(*vl)
        plt.plot(e, l, "-", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{ds_key}: Training and Validation Loss Curves")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_key}_loss_curves.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------- Plot 2: validation metrics ----------------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    plt.figure()
    metrics = experiment_data[ds_key]["metrics"]["val"]  # (epoch,cwa,swa,hm,ocga)
    if metrics:
        ep = [t[0] for t in metrics]
        cwa = [t[1] for t in metrics]
        swa = [t[2] for t in metrics]
        hm = [t[3] for t in metrics]
        ocg = [t[4] for t in metrics]
        plt.plot(ep, cwa, label="CWA")
        plt.plot(ep, swa, label="SWA")
        plt.plot(ep, hm, label="HM")
        plt.plot(ep, ocg, label="OCGA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{ds_key}: Validation Metrics")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_key}_validation_metrics.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# -------- Plot 3: HM with best epoch marker ---------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    plt.figure()
    metrics = experiment_data[ds_key]["metrics"]["val"]
    if metrics:
        ep = np.array([t[0] for t in metrics])
        hm = np.array([t[3] for t in metrics])
        plt.plot(ep, hm, label="HM")
        best_idx = hm.argmax()
        plt.scatter(
            ep[best_idx],
            hm[best_idx],
            color="red",
            zorder=5,
            label=f"Best@{int(ep[best_idx])}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Mean (HM)")
    plt.title(f"{ds_key}: Validation HM and Best Epoch")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_key}_validation_HM.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating HM plot: {e}")
    plt.close()

# -------- Plot 4: confusion matrix ------------------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    y_true = np.array(experiment_data[ds_key]["ground_truth"])
    y_pred = np.array(experiment_data[ds_key]["predictions"])
    if y_true.size == 0 or y_true.size != y_pred.size:
        raise ValueError("Predictions / ground truth missing or mismatched.")
    n_cls = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_cls, n_cls), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{ds_key}: Test Confusion Matrix")
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
            )
    fname = os.path.join(working_dir, f"{ds_key}_confusion_matrix.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
