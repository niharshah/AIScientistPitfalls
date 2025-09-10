import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

saved = []
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data["MultiSynGen"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}


# helper to pull lists safely
def col(lst, idx):
    return [t[idx] for t in lst]


# 1) train / val loss curves
try:
    plt.figure()
    for ds, rec in runs.items():
        ep = col(rec["losses"]["train"], 0)
        plt.plot(ep, col(rec["losses"]["train"], 1), label=f"{ds}_train")
        plt.plot(ep, col(rec["losses"]["val"], 1), "--", label=f"{ds}_val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss (MultiSynGen)", loc="center")
    plt.legend()
    fname = os.path.join(working_dir, "MultiSynGen_loss_curves.png")
    plt.savefig(fname)
    saved.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) validation SWA
try:
    plt.figure()
    for ds, rec in runs.items():
        ep = col(rec["metrics"]["val"], 0)
        plt.plot(ep, col(rec["metrics"]["val"], 1), label=ds)
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("Validation SWA over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "MultiSynGen_val_SWA.png")
    plt.savefig(fname)
    saved.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# 3) validation ZRGS
try:
    plt.figure()
    for ds, rec in runs.items():
        ep = col(rec["metrics"]["val"], 0)
        plt.plot(ep, col(rec["metrics"]["val"], 3), label=ds)
    plt.xlabel("Epoch")
    plt.ylabel("Zero-shot Rule-Gen Score")
    plt.title("Validation ZRGS over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "MultiSynGen_val_ZRGS.png")
    plt.savefig(fname)
    saved.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating ZRGS plot: {e}")
    plt.close()

# 4) test SWA self vs holdout
try:
    plt.figure()
    ds_names = list(runs.keys())
    x = np.arange(len(ds_names))
    width = 0.35
    swa_self = [runs[d]["metrics"]["self_test"][0] for d in ds_names]
    swa_hold = [runs[d]["metrics"]["holdout_test"][0] for d in ds_names]
    plt.bar(x - width / 2, swa_self, width, label="Self-test")
    plt.bar(x + width / 2, swa_hold, width, label="Hold-out")
    plt.xticks(x, ds_names)
    plt.ylim(0, 1)
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("Self vs Hold-out Test SWA")
    plt.legend()
    fname = os.path.join(working_dir, "MultiSynGen_test_SWA_bar.png")
    plt.savefig(fname)
    saved.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test bar plot: {e}")
    plt.close()

print("Saved figures:", saved)
