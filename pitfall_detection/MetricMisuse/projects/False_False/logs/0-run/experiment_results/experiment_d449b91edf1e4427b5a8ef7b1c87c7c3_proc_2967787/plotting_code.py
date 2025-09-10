import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]

    # ------------ gather arrays -------------
    pre_ls = np.array(data["losses"].get("pretrain", []))
    tr_ls = np.array(data["losses"].get("train", []))
    val_ls = np.array(data["losses"].get("val", []))

    swa = np.array(data["metrics"].get("val_SWA", []))
    cwa = np.array(data["metrics"].get("val_CWA", []))
    scwa = np.array(data["metrics"].get("val_SCWA", []))

    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))
    acc = (preds == gts).mean() if len(preds) else float("nan")

    # --------------- Plot 1: loss curves ----------------
    try:
        plt.figure()
        epochs_pre = np.arange(1, len(pre_ls) + 1)
        epochs_cls = np.arange(1, len(tr_ls) + 1)
        plt.plot(epochs_pre, pre_ls, label="Pre-train Loss")
        if len(tr_ls):
            plt.plot(epochs_cls, tr_ls, label="Train Loss")
        if len(val_ls):
            plt.plot(epochs_cls, val_ls, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss over Epochs\nLeft: Pre-train, Right: Fine-tune")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # --------------- Plot 2: metric curves --------------
    try:
        plt.figure()
        ep_m = np.arange(1, len(swa) + 1)
        if len(swa):
            plt.plot(ep_m, swa, label="SWA")
        if len(cwa):
            plt.plot(ep_m, cwa, label="CWA")
        if len(scwa):
            plt.plot(ep_m, scwa, label="SCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH Validation Metrics over Epochs\nLeft: SWA, CWA, SCWA")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_metric_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot: {e}")
        plt.close()

    # --------------- Plot 3: final bars -----------------
    try:
        plt.figure()
        final_vals = [
            swa[-1] if len(swa) else 0,
            cwa[-1] if len(cwa) else 0,
            scwa[-1] if len(scwa) else 0,
            acc,
        ]
        names = ["SWA", "CWA", "SCWA", "Accuracy"]
        plt.bar(names, final_vals, color="skyblue")
        for i, v in enumerate(final_vals):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Final Validation Scores\nRight: Overall Accuracy")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_final_scores.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final score plot: {e}")
        plt.close()

    # ----------------- print summary -------------------
    print(f"Final Accuracy: {acc:.4f}")
    if len(swa):
        print(f"Final SWA : {swa[-1]:.4f}")
    if len(cwa):
        print(f"Final CWA : {cwa[-1]:.4f}")
    if len(scwa):
        print(f"Final SCWA: {scwa[-1]:.4f}")
