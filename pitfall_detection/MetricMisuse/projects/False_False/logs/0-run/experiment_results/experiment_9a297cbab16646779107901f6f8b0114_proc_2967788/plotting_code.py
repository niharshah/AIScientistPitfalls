import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load experiment data ---------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_contrastive" in experiment_data:
    data = experiment_data["SPR_contrastive"]

    # ---------- pull losses ----------
    pre_loss = np.array(data["losses"].get("pretrain", []))
    tr_loss = np.array(data["losses"].get("train", []))
    val_loss = np.array(data["losses"].get("val", []))
    epochs_fine = np.arange(1, len(tr_loss) + 1)
    epochs_pre = np.arange(-len(pre_loss) + 1, 1)  # negative indices for pre-train

    # ------------------ Plot 1: Loss curves -------------------------
    try:
        plt.figure()
        if pre_loss.size:
            plt.plot(epochs_pre, pre_loss, label="Pre-train Loss")
        if tr_loss.size:
            plt.plot(epochs_fine, tr_loss, label="Train Loss")
        if val_loss.size:
            plt.plot(epochs_fine, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("NT-Xent / CE Loss")
        plt.title(
            "SPR_contrastive Loss over Epochs\n"
            "Left: Pre-train, Center: Train, Right: Val"
        )
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_contrastive_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- pull metric curves ----------
    swa = np.array(data["metrics"].get("val_SWA", []))
    cwa = np.array(data["metrics"].get("val_CWA", []))
    scwa = np.array(data["metrics"].get("val_SCWA", []))
    epochs_metric = np.arange(1, len(swa) + 1)

    # ------------------ Plot 2: Metric curves -----------------------
    try:
        plt.figure()
        if swa.size:
            plt.plot(epochs_metric, swa, label="SWA")
        if cwa.size:
            plt.plot(epochs_metric, cwa, label="CWA")
        if scwa.size:
            plt.plot(epochs_metric, scwa, label="SCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title(
            "SPR_contrastive Validation Metrics over Epochs\n"
            "Left: SWA, Middle: CWA, Right: SCWA"
        )
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_contrastive_metric_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric curve: {e}")
        plt.close()

    # ------------------ Plot 3: Final metric bar --------------------
    try:
        finals = [
            swa[-1] if swa.size else 0,
            cwa[-1] if cwa.size else 0,
            scwa[-1] if scwa.size else 0,
        ]
        labels = ["SWA", "CWA", "SCWA"]
        plt.figure()
        plt.bar(labels, finals, color="lightgreen")
        plt.ylabel("Final Validation Score")
        plt.title("SPR_contrastive Final Validation Metrics")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_contrastive_final_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final metric bar plot: {e}")
        plt.close()

    # ------------------ Print best SCWA -----------------------------
    if scwa.size:
        best_idx = scwa.argmax()
        print(f"Best epoch for SCWA: {best_idx+1} | " f"SCWA={scwa[best_idx]:.4f}")
