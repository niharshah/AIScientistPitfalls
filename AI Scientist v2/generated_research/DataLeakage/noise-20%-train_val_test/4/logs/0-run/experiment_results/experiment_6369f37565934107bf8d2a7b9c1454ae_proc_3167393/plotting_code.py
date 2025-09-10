import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# we will look at at most the first 5 datasets
datasets = list(experiment_data.keys())[:5]


# ---------- helper to fetch nested arrays ----------
def safe_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return np.asarray(cur)


# ---------- 1) Loss curves ----------
try:
    for ds in datasets:
        epochs = safe_get(experiment_data[ds], "epochs", default=[])
        train_loss = safe_get(experiment_data[ds], "losses", "train", default=[])
        val_loss = safe_get(experiment_data[ds], "losses", "val", default=[])
        if len(epochs) == 0:
            continue  # nothing to plot

        plt.figure(figsize=(10, 4))
        # Left: train loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label="train")
        plt.title(f"Left: Training Loss - {ds}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        # Right: val loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_loss, label="val", color="orange")
        plt.title(f"Right: Validation Loss - {ds}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds}_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- 2) F1 curves ----------
try:
    for ds in datasets:
        epochs = safe_get(experiment_data[ds], "epochs", default=[])
        train_f1 = safe_get(experiment_data[ds], "metrics", "train_f1", default=[])
        val_f1 = safe_get(experiment_data[ds], "metrics", "val_f1", default=[])
        if len(epochs) == 0:
            continue
        plt.figure(figsize=(10, 4))
        # Left: train F1
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_f1, label="train")
        plt.title(f"Left: Training Macro-F1 - {ds}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        # Right: val F1
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_f1, label="val", color="green")
        plt.title(f"Right: Validation Macro-F1 - {ds}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds}_f1_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# ---------- 3) Test F1 bar comparing datasets ----------
try:
    plt.figure()
    test_f1s, labels = [], []
    for ds in datasets:
        tf = safe_get(experiment_data[ds], "metrics", "test_f1", default=[None])
        tf = tf[0] if isinstance(tf, (list, np.ndarray)) else tf
        if tf is not None:
            test_f1s.append(tf)
            labels.append(ds)
            print(f"{ds} Test Macro-F1 = {tf:.4f}")
    if test_f1s:
        plt.bar(range(len(test_f1s)), test_f1s, tick_label=labels)
        plt.title("Test Macro-F1 across datasets")
        plt.ylabel("Macro-F1")
        fname = os.path.join(working_dir, "all_datasets_test_f1_bar.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating test F1 bar plot: {e}")
    plt.close()
