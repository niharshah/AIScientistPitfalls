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
    ed = experiment_data["bag_of_emb"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed is not None:
    # convenience vars
    tr_loss, va_loss = ed["losses"]["train"], ed["losses"]["val"]
    tr_acc, va_acc = ed["metrics"]["train"], ed["metrics"]["val"]
    tr_swa, va_swa = ed["swa"]["train"], ed["swa"]["val"]

    # ---------- plot 1 : loss ----------
    try:
        plt.figure()
        epochs = range(1, len(tr_loss) + 1)
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, va_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "Loss Curves\nLeft: Training, Right: Validation  (Dataset: SPR_BENCH)"
        )
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- plot 2 : accuracy ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_acc, label="Train")
        plt.plot(epochs, va_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            "Accuracy Curves\nLeft: Training, Right: Validation  (Dataset: SPR_BENCH)"
        )
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------- plot 3 : shape-weighted accuracy ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_swa, label="Train")
        plt.plot(epochs, va_swa, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SWA Curves\nLeft: Training, Right: Validation  (Dataset: SPR_BENCH)")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_swa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating swa plot: {e}")
        plt.close()

    # ---------- print final test metrics ----------
    tmet = ed.get("test_metrics", {})
    print(
        f"Test metrics -> loss: {tmet.get('loss'):.4f}, acc: {tmet.get('acc'):.3f}, swa: {tmet.get('swa'):.3f}"
    )
