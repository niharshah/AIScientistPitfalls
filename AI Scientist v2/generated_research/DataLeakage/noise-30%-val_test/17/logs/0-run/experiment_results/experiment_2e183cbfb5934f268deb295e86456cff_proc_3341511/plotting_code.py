import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import matthews_corrcoef, f1_score

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

exp = experiment_data["Constant_LR"]["SPR_BENCH"]
tr_loss, va_loss = exp["losses"]["train"], exp["losses"]["val"]
tr_mcc, va_mcc = exp["metrics"]["train"], exp["metrics"]["val"]
preds_all, gts_all, cfgs = exp["predictions"], exp["ground_truth"], exp["configs"]

# ---------- iterate through runs ----------
ptr = 0
for i, cfg in enumerate(cfgs):
    ep = cfg["epochs"]
    tl = tr_loss[ptr : ptr + ep]
    vl = va_loss[ptr : ptr + ep]
    tm = tr_mcc[ptr : ptr + ep]
    vm = va_mcc[ptr : ptr + ep]
    ptr += ep

    # ---- plotting ----
    try:
        fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        # Loss subplot
        ax[0].plot(range(1, ep + 1), tl, label="Train")
        ax[0].plot(range(1, ep + 1), vl, label="Val")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("BCE Loss")
        ax[0].set_title("Loss")
        ax[0].legend()
        # MCC subplot
        ax[1].plot(range(1, ep + 1), tm, label="Train")
        ax[1].plot(range(1, ep + 1), vm, label="Val")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("MCC")
        ax[1].set_title("Matthew CorrCoef")
        ax[1].legend()
        fig.suptitle(f'Run {i+1} - SPR_BENCH (Constant LR={cfg["lr"]})', fontsize=14)
        save_path = os.path.join(working_dir, f"SPR_BENCH_Run{i+1}_Loss_MCC.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)
    except Exception as e:
        print(f"Error creating plot for run {i+1}: {e}")
    finally:
        plt.close(fig)

    # ---- evaluation metrics ----
    try:
        mcc = matthews_corrcoef(gts_all[i], preds_all[i])
        f1 = f1_score(gts_all[i], preds_all[i], average="macro")
        print(f"Run {i+1} | Test MCC: {mcc:.4f} | Test Macro-F1: {f1:.4f}")
    except Exception as e:
        print(f"Error computing metrics for run {i+1}: {e}")
