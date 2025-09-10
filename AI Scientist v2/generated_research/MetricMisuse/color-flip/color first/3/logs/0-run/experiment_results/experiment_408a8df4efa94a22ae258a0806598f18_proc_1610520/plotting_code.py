import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------- #
# 1. Load experiment data                                     #
# ----------------------------------------------------------- #
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = exp.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = {}


# helpers
def unpack(list_of_tuples, idx):
    return [t[idx] for t in list_of_tuples]


# ----------------------------------------------------------- #
# 2. Plot: train / val loss curves                            #
# ----------------------------------------------------------- #
try:
    plt.figure()
    tr_epochs = unpack(run["losses"]["train"], 0)
    tr_loss = unpack(run["losses"]["train"], 1)
    val_epochs = unpack(run["losses"]["val"], 0)
    val_loss = unpack(run["losses"]["val"], 1)
    plt.plot(tr_epochs, tr_loss, "--", label="train")
    plt.plot(val_epochs, val_loss, "-", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("SPR_BENCH: Train vs. Val Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ----------------------------------------------------------- #
# 3. Plot: validation metric curves (CWA, SWA, SNWA)          #
# ----------------------------------------------------------- #
try:
    plt.figure()
    val_epochs = unpack(run["metrics"]["val"], 0)
    cwa = unpack(run["metrics"]["val"], 1)
    swa = unpack(run["metrics"]["val"], 2)
    snwa = unpack(run["metrics"]["val"], 3)
    plt.plot(val_epochs, cwa, label="CWA")
    plt.plot(val_epochs, swa, label="SWA")
    plt.plot(val_epochs, snwa, label="SNWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Validation Metrics")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_metrics.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating metric curves plot: {e}")
    plt.close()

# ----------------------------------------------------------- #
# 4. Plot: best SNWA bar (single run)                         #
# ----------------------------------------------------------- #
try:
    best_snwa = max(snwa) if snwa else 0
    best_ep = val_epochs[int(np.argmax(snwa))] if snwa else 0
    plt.figure()
    plt.bar(["best"], [best_snwa])
    plt.ylabel("SNWA")
    plt.title(f"SPR_BENCH: Best Dev SNWA (epoch {best_ep})")
    fname = os.path.join(working_dir, "SPR_BENCH_best_SNWA_bar.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating best SNWA bar chart: {e}")
    plt.close()

# ----------------------------------------------------------- #
# 5. Confusion matrices (dev & test)                          #
# ----------------------------------------------------------- #
for split in ["dev", "test"]:
    try:
        gts = run["ground_truth"][split]
        preds = run["predictions"][split]
        cm = confusion_matrix(gts, preds)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title(f"SPR_BENCH {split.capitalize()} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        fname = os.path.join(working_dir, f"SPR_BENCH_{split}_confusion_matrix.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating {split} confusion matrix: {e}")
        plt.close()
