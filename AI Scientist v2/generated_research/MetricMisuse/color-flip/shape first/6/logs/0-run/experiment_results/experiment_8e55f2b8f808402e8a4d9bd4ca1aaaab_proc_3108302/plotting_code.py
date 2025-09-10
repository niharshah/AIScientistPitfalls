import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data["contrastive_context_aware"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = {}


# ---------- helper ----------
def arr(dic, *keys):
    tmp = dic
    for k in keys:
        tmp = tmp[k]
    return np.asarray(tmp)


epochs = np.arange(1, len(run.get("epochs", [])) + 1)

# ---------- 1. loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, arr(run, "losses", "train"), "r--", label="Train Loss")
    plt.plot(epochs, arr(run, "losses", "val"), "b-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss Curves\nDataset: contrastive_context_aware (Train vs. Validation)")
    plt.legend()
    fname = os.path.join(working_dir, "contrastive_context_aware_loss_curves.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- 2. SWA ----------
try:
    plt.figure()
    plt.plot(epochs, arr(run, "metrics", "val_swa"), "g-o")
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    plt.title("Validation SWA Across Epochs\nDataset: contrastive_context_aware")
    fname = os.path.join(working_dir, "contrastive_context_aware_val_SWA.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# ---------- 3. CWA ----------
try:
    plt.figure()
    plt.plot(epochs, arr(run, "metrics", "val_cwa"), "m-o")
    plt.xlabel("Epoch")
    plt.ylabel("CWA")
    plt.title("Validation CWA Across Epochs\nDataset: contrastive_context_aware")
    fname = os.path.join(working_dir, "contrastive_context_aware_val_CWA.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# ---------- 4. SCAA ----------
try:
    plt.figure()
    plt.plot(epochs, arr(run, "metrics", "val_scaa"), "c-o")
    plt.xlabel("Epoch")
    plt.ylabel("SCAA")
    plt.title("Validation SCAA Across Epochs\nDataset: contrastive_context_aware")
    fname = os.path.join(working_dir, "contrastive_context_aware_val_SCAA.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating SCAA plot: {e}")
    plt.close()

# ---------- print best metrics ----------
if run:
    best_swa = arr(run, "metrics", "val_swa").max()
    best_cwa = arr(run, "metrics", "val_cwa").max()
    best_scaa = arr(run, "metrics", "val_scaa").max()
    print(f"Best Validation SWA : {best_swa :.3f}")
    print(f"Best Validation CWA : {best_cwa :.3f}")
    print(f"Best Validation SCAA: {best_scaa:.3f}")
