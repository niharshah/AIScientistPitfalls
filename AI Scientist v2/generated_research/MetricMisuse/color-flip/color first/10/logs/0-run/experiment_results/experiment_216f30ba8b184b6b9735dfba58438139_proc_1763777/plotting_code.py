import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data -----------------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    (exp,) = [{}]  # dummy to avoid NameError

# Safely dig into dict
try:
    ablt = next(iter(exp))
    dset = next(iter(exp[ablt]))
    md = exp[ablt][dset]["metrics"]
    preds = np.array(exp[ablt][dset]["predictions"])
    gts = np.array(exp[ablt][dset]["ground_truth"])
except Exception as e:
    print(f"Experiment structure issue: {e}")
    md, preds, gts = {}, np.array([]), np.array([])

epochs = np.arange(1, len(md.get("train_loss", [])) + 1)

# ---------- 1) train/val loss -------------
try:
    plt.figure()
    plt.plot(epochs, md["train_loss"], label="Train Loss")
    plt.plot(epochs, md["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(f"{dset} — Train vs Val Loss\nAblation: {ablt}")
    plt.legend()
    fname = os.path.join(working_dir, f"{dset}_{ablt}_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- 2) val CWA --------------------
try:
    plt.figure()
    plt.plot(epochs, md["val_CWA"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("CWA")
    plt.title(f"{dset} — Color-Weighted Accuracy\nAblation: {ablt}")
    fname = os.path.join(working_dir, f"{dset}_{ablt}_CWA_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# ---------- 3) val SWA --------------------
try:
    plt.figure()
    plt.plot(epochs, md["val_SWA"], marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    plt.title(f"{dset} — Shape-Weighted Accuracy\nAblation: {ablt}")
    fname = os.path.join(working_dir, f"{dset}_{ablt}_SWA_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# ---------- 4) val CWA2 -------------------
try:
    plt.figure()
    plt.plot(epochs, md["val_CWA2"], marker="o", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("CWA²")
    plt.title(f"{dset} — Complexity-Weighted Accuracy\nAblation: {ablt}")
    fname = os.path.join(working_dir, f"{dset}_{ablt}_CWA2_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CWA2 plot: {e}")
    plt.close()

# ---------- 5) confusion matrix -----------
try:
    tp = int(((preds == 1) & (gts == 1)).sum())
    fp = int(((preds == 1) & (gts == 0)).sum())
    tn = int(((preds == 0) & (gts == 0)).sum())
    fn = int(((preds == 0) & (gts == 1)).sum())
    plt.figure()
    plt.bar(
        ["TP", "FP", "TN", "FN"],
        [tp, fp, tn, fn],
        color=["blue", "orange", "green", "red"],
    )
    plt.ylabel("Count")
    plt.title(f"{dset} — Confusion Matrix Counts\nAblation: {ablt}")
    fname = os.path.join(working_dir, f"{dset}_{ablt}_confusion_counts.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

print("Finished plotting. Files saved to", working_dir)
