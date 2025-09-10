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
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("epochs_tuning", {})

# ---------- iterate runs ----------
for run_key, run_data in runs.items():
    # ---------- loss curve ----------
    try:
        plt.figure()
        plt.plot(run_data["losses"]["train"], label="train")
        plt.plot(run_data["losses"]["val"], label="val")
        plt.title(f"SPR_BENCH | {run_key} | Cross-Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = f"SPR_BENCH_{run_key}_loss.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {run_key}: {e}")
        plt.close()

    # ---------- accuracy curve ----------
    try:
        plt.figure()
        plt.plot(run_data["metrics"]["train"], label="train")
        plt.plot(run_data["metrics"]["val"], label="val")
        plt.title(f"SPR_BENCH | {run_key} | Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = f"SPR_BENCH_{run_key}_accuracy.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {run_key}: {e}")
        plt.close()

    # ---------- print final val accuracy ----------
    try:
        final_acc = (
            run_data["metrics"]["val"][-1] if run_data["metrics"]["val"] else None
        )
        print(f"Run {run_key} final val accuracy: {final_acc:.4f}")
    except Exception as e:
        print(f"Error computing final accuracy for {run_key}: {e}")
