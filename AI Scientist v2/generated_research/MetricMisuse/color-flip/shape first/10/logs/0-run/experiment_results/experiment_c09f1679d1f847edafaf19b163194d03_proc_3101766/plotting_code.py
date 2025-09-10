import matplotlib.pyplot as plt
import numpy as np
import os

# prepare working directory ----------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data ---------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

if "SPR" not in experiment_data:
    print("Dataset 'SPR' not found in experiment_data.npy")
    raise SystemExit
bench = experiment_data["SPR"]

# helper arrays ----------------------------------------------------------------
train_loss = np.array(bench["losses"]["train"])
val_loss = np.array(bench["losses"]["val"])

# metrics over epochs
val_metrics = bench["metrics"]["val"]
swa = np.array([m.get("swa", np.nan) for m in val_metrics])
cwa = np.array([m.get("cwa", np.nan) for m in val_metrics])
acs = np.array([m.get("acs", np.nan) for m in val_metrics])

epochs = np.arange(1, len(train_loss) + 1)

# 1) Loss curves ---------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR: Training vs. Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_loss_curves.png")
    plt.savefig(fname, dpi=150)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Weighted accuracy curves --------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, acs, label="ACS")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR: Validation Weighted Accuracies")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_weighted_accuracy_curves.png")
    plt.savefig(fname, dpi=150)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# 3) Test accuracy bar ---------------------------------------------------------
try:
    preds = np.array(bench["predictions"])
    gts = np.array(bench["ground_truth"])
    if preds.size and gts.size:
        acc = (preds == gts).mean()
    else:
        acc = np.nan
    plt.figure()
    plt.bar([0], [acc])
    plt.ylim(0, 1.0)
    plt.xticks([0], ["SPR"])
    plt.ylabel("Accuracy")
    plt.title("SPR: Test Accuracy")
    fname = os.path.join(working_dir, "SPR_test_accuracy.png")
    plt.savefig(fname, dpi=150)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy plot: {e}")
    plt.close()

print(f"Final SPR test accuracy: {acc:.3f}")
