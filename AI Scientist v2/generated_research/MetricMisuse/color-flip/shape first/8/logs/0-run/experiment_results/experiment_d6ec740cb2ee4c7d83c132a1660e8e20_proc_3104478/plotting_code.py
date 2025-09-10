import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# locate run & dataset safely
run_name = "NoAugContrastive"
dset_name = "SPR_BENCH"
ed = experiment_data.get(run_name, {}).get(dset_name, {})

train_loss = ed.get("losses", {}).get("train", [])
val_loss = ed.get("losses", {}).get("val", [])
val_ccwa = ed.get("metrics", {}).get("val_CCWA", [])
preds_list = ed.get("predictions", [])
gts_list = ed.get("ground_truth", [])

# 1) Loss curves -------------------------------------------------------------
try:
    if train_loss and val_loss:
        epochs = np.arange(1, len(train_loss) + 1)
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs. Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2) CCWA curve --------------------------------------------------------------
try:
    if val_ccwa:
        epochs = np.arange(1, len(val_ccwa) + 1)
        plt.figure()
        plt.plot(epochs, val_ccwa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("CCWA")
        plt.title("SPR_BENCH: Validation CCWA over Epochs")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_CCWA_curve.png"))
        plt.close()
except Exception as e:
    print(f"Error creating CCWA plot: {e}")
    plt.close()

# 3) Confusion matrix for last epoch ----------------------------------------
try:
    if preds_list and gts_list:
        preds = np.array(preds_list[-1])
        gts = np.array(gts_list[-1])
        n_cls = max(np.max(preds), np.max(gts)) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix (final epoch)")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# Print final metric for quick inspection
if val_ccwa:
    print(f"Final validation CCWA: {val_ccwa[-1]:.4f}")
