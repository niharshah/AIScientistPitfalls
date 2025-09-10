import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

best_tag, best_cwa = None, -1.0

# ---------- iterate & plot ----------
for tag, run in experiment_data.get("embedding_dim", {}).items():
    try:
        tr_loss = run["losses"]["train"]
        val_loss = run["losses"]["val"]
        cwa_val = run["metrics"]["val"]
        epochs = np.arange(1, len(tr_loss) + 1)

        plt.figure(figsize=(10, 4))

        # Left: loss curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()

        # Right: validation CWA
        plt.subplot(1, 2, 2)
        plt.plot(epochs, cwa_val, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("CWA-2D")
        plt.title("Complexity Weighted Accuracy")

        plt.suptitle(f"SPR_BENCH {tag}\nLeft: Loss, Right: Validation CWA")
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])

        fname = os.path.join(working_dir, f"{tag}_training_curves.png")
        plt.savefig(fname)
        plt.close()

        if cwa_val and max(cwa_val) > best_cwa:
            best_cwa, best_tag = max(cwa_val), tag
        print(f"{tag}: Final Val CWA = {cwa_val[-1]:.4f}")
    except Exception as e:
        print(f"Error creating plot for {tag}: {e}")
        plt.close()

# ---------- summary ----------
if best_tag is not None:
    print(f"Best run: {best_tag} with max Val CWA {best_cwa:.4f}")
