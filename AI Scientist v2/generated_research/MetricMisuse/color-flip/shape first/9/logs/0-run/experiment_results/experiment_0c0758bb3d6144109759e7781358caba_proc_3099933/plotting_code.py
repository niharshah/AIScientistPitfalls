import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    spr_data = experiment_data["num_epochs_sweep"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = None

if spr_data:
    # ---------- metrics ----------
    preds = np.array(spr_data.get("predictions", []))
    gtruth = np.array(spr_data.get("ground_truth", []))
    acc = (preds == gtruth).mean() if preds.size else float("nan")
    best_cwa = max(spr_data.get("best_val_metric", [float("nan")]))
    print(f"Overall accuracy (best run): {acc:.4f}")
    print(f"Best validation CWA-2D (across sweeps): {best_cwa:.4f}")

    # ---------- 1. loss curves ----------
    try:
        plt.figure()
        epochs = np.arange(1, len(spr_data["losses"]["train"]) + 1)
        plt.plot(epochs, spr_data["losses"]["train"], label="train")
        plt.plot(epochs, spr_data["losses"]["val"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH – Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- 2. CWA vs max_epochs ----------
    try:
        plt.figure()
        x = spr_data["config_epochs"]
        y = spr_data["best_val_metric"]
        plt.bar(x, y, color="skyblue")
        plt.xlabel("max_epochs setting")
        plt.ylabel("Best Validation CWA-2D")
        plt.title("SPR_BENCH – CWA-2D versus max_epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_CWA_vs_max_epochs.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CWA bar plot: {e}")
        plt.close()

    # ---------- 3. confusion matrix ----------
    try:
        if preds.size and gtruth.size:
            num_cls = int(max(preds.max(), gtruth.max())) + 1
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(gtruth, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.ylabel("Ground Truth label")
            plt.xlabel("Predicted label")
            plt.title("SPR_BENCH – Confusion Matrix\n(rows = GT, cols = Pred)")
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()
