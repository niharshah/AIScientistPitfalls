import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR" in experiment_data:
    data = experiment_data["SPR"]

    pre_ls = np.asarray(data["losses"].get("pretrain", []), dtype=float)
    tr_ls = np.asarray(data["losses"].get("train", []), dtype=float)
    val_ls = np.asarray(data["losses"].get("val", []), dtype=float)
    swa_hist = np.asarray(data["metrics"].get("SWA", []), dtype=float)
    cwa_hist = np.asarray(data["metrics"].get("CWA", []), dtype=float)
    sc_hist = np.asarray(data["metrics"].get("SCWA", []), dtype=float)
    preds = np.asarray(data.get("predictions", []), dtype=int)
    gts = np.asarray(data.get("ground_truth", []), dtype=int)

    # ---------------- Plot 1: pre-training loss -----------------
    try:
        if pre_ls.size:
            plt.figure()
            plt.plot(np.arange(1, pre_ls.size + 1), pre_ls, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("NT-Xent Loss")
            plt.title("SPR Pre-training Loss Curve")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_pretrain_loss.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating pretrain loss plot: {e}")
        plt.close()

    # --------------- Plot 2: train vs val loss ------------------
    try:
        if tr_ls.size and val_ls.size:
            epochs = np.arange(1, tr_ls.size + 1)
            plt.figure()
            plt.plot(epochs, tr_ls, label="Train Loss")
            plt.plot(epochs, val_ls, label="Val Loss")
            plt.xlabel("Fine-tune Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("SPR Loss over Epochs\nLeft: Train, Right: Val")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_train_val_loss.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating train/val loss plot: {e}")
        plt.close()

    # --------------- Plot 3: metrics curves ---------------------
    try:
        if sc_hist.size:
            epochs = np.arange(1, sc_hist.size + 1)
            plt.figure()
            if swa_hist.size:
                plt.plot(epochs, swa_hist, label="SWA")
            if cwa_hist.size:
                plt.plot(epochs, cwa_hist, label="CWA")
            plt.plot(epochs, sc_hist, label="SCWA")
            plt.xlabel("Fine-tune Epoch")
            plt.ylabel("Score")
            plt.title("SPR Evaluation Metrics over Epochs")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_metrics_curve.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()

    # --------------- Plot 4: confusion matrix -------------------
    try:
        if preds.size and gts.size and preds.shape == gts.shape:
            classes = sorted(set(gts.tolist() + preds.tolist()))
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("SPR Confusion Matrix (Best SCWA Model)")
            plt.xticks(classes)
            plt.yticks(classes)
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------------- print summary metrics ---------------------
    if sc_hist.size:
        print(
            f"Final epoch metrics -> SWA: {swa_hist[-1]:.4f}, "
            f"CWA: {cwa_hist[-1]:.4f}, SCWA: {sc_hist[-1]:.4f}"
        )
    if "best_scwa" in data or (sc_hist.size and sc_hist.max()):
        best_scwa = data.get("best_scwa", float(sc_hist.max()))
        print(f"Best SCWA achieved: {best_scwa:.4f}")
else:
    print("SPR data not found in experiment_data.")
