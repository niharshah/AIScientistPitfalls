import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up working dir ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ---------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- collect test scores ----------------------------------------------
test_scores = {}
for dset, sub in experiment_data.items():
    tst = sub["metrics"].get("test_cplxwa")
    if tst is not None:
        test_scores[dset] = tst

# ---------- per-dataset plots -------------------------------------------------
for dset, sub in experiment_data.items():
    epochs = sub.get("epochs", [])
    tr_loss = sub["losses"].get("train", [])
    va_loss = sub["losses"].get("val", [])
    tr_cplx = sub["metrics"].get("train_cplxwa", [])
    va_cplx = sub["metrics"].get("val_cplxwa", [])

    # ---- Loss curve ----------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, va_loss, label="Val Loss")
        plt.title(f"{dset} Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset.lower()}_loss_curve.png")
        plt.savefig(fname)
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
    finally:
        plt.close()

    # ---- CompWA curve --------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_cplx, label="Train CompWA")
        plt.plot(epochs, va_cplx, label="Val CompWA")
        plt.title(f"{dset} Complexity-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("CompWA")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset.lower()}_compwa_curve.png")
        plt.savefig(fname)
    except Exception as e:
        print(f"Error creating CompWA plot for {dset}: {e}")
    finally:
        plt.close()

# ---------- summary bar plot --------------------------------------------------
try:
    if test_scores:
        plt.figure()
        names, vals = zip(*test_scores.items())
        plt.bar(names, vals)
        plt.title("Test Complexity-Weighted Accuracy Comparison")
        plt.xlabel("Dataset")
        plt.ylabel("Test CompWA")
        fname = os.path.join(working_dir, "summary_test_compwa.png")
        plt.savefig(fname)
    else:
        print("No test CompWA data found.")
except Exception as e:
    print(f"Error creating summary bar plot: {e}")
finally:
    plt.close()

# ---------- print evaluation metrics -----------------------------------------
for dset, score in test_scores.items():
    print(f"{dset}: Test CompWA = {score:.4f}")
