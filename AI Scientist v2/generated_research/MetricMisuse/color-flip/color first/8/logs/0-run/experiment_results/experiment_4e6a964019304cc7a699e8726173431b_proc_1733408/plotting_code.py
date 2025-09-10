import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data ----------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None


# -------------------- helper: safe extraction --------------------------------
def unzip(pairs):
    if not pairs:
        return [], []
    a, b = zip(*pairs)
    return list(a), list(b)


# -------------------- Plot 1: loss curves ------------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    plt.figure()
    for dname, dct in experiment_data.items():
        tr_epochs, tr_losses = unzip(dct["losses"]["train"])
        if tr_epochs:
            plt.plot(tr_epochs, tr_losses, "--", label=f"{dname}-train")
        val_epochs, val_losses = unzip(dct["losses"]["val"])
        if val_epochs:
            plt.plot(val_epochs, val_losses, "-", label=f"{dname}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training vs Validation Loss Curves")
    plt.legend()
    path = os.path.join(working_dir, "loss_curves_all_datasets.png")
    plt.savefig(path)
    print(f"Saved {path}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# -------------------- Plot 2/3/4: validation metrics -------------------------
metric_idx = {"CWA": 1, "SWA": 2, "HM": 3}
for mname, midx in metric_idx.items():
    try:
        if experiment_data is None:
            raise ValueError("No experiment data loaded.")
        plt.figure()
        for dname, dct in experiment_data.items():
            vals = dct["metrics"]["val"]  # (epoch,cwa,swa,hm,ocga)
            if not vals:
                continue
            epochs = [t[0] for t in vals]
            scores = [t[midx] for t in vals]
            plt.plot(epochs, scores, label=dname)
        plt.xlabel("Epoch")
        plt.ylabel(mname)
        plt.title(f"Validation {mname} Across Datasets")
        plt.legend()
        fname = os.path.join(working_dir, f"val_{mname.lower()}_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating {mname} plot: {e}")
        plt.close()

# -------------------- Plot 5: confusion matrix -------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    plotted = 0
    for dname, dct in experiment_data.items():
        if plotted >= 2:  # keep total figure count ≤5
            break
        preds = np.array(dct.get("predictions", []))
        gts = np.array(dct.get("ground_truth", []))
        if preds.size == 0 or gts.size == 0:
            continue
        n_cls = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{dname}: Confusion Matrix (Test)")
        plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
        plt.close()
        plotted += 1
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
