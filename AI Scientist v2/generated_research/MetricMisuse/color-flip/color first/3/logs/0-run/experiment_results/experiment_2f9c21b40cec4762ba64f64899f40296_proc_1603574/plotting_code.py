import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# Load experiment data                                               #
# ------------------------------------------------------------------ #
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    lr_dict = exp["learning_rate"]["SPR_BENCH"]
    best_lr = lr_dict["best_lr"]
    lrs = list(lr_dict["losses"]["train"].keys())

    # -------------------------------------------------------------- #
    # 1. Train / Val loss curves for best lr                         #
    # -------------------------------------------------------------- #
    try:
        train_loss = lr_dict["losses"]["train"][best_lr]
        val_loss = lr_dict["losses"]["val"][best_lr]
        epochs_t = [e for e, _ in train_loss]
        train_vals = [v for _, v in train_loss]
        val_vals = [v for _, v in val_loss]

        plt.figure()
        plt.plot(epochs_t, train_vals, label="Train Loss")
        plt.plot(epochs_t, val_vals, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH Loss Curves (lr={best_lr})")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_loss_curves_lr_{best_lr}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # 2. Validation metrics curves for best lr                       #
    # -------------------------------------------------------------- #
    try:
        val_metrics = lr_dict["metrics"]["val"][best_lr]
        epochs_m = [e for e, *_ in val_metrics]
        cwa_vals = [c for _, c, _, _ in val_metrics]
        swa_vals = [s for *_, s, _ in val_metrics]  # type: ignore
        hcs_vals = [h for *_, h in val_metrics]  # type: ignore

        plt.figure()
        plt.plot(epochs_m, cwa_vals, label="CWA")
        plt.plot(epochs_m, swa_vals, label="SWA")
        plt.plot(epochs_m, hcs_vals, label="HCSA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"SPR_BENCH Validation Metrics (lr={best_lr})")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_metrics_curves_lr_{best_lr}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric curve plot: {e}")
        plt.close()

    # -------------------------------------------------------------- #
    # 3. Final dev HCSA for each lr                                  #
    # -------------------------------------------------------------- #
    try:
        final_hcs = []
        for lr in lrs:
            dev_preds = lr_dict["predictions"]["dev"][lr]
            dev_gts = lr_dict["ground_truth"]["dev"][lr]
            # recompute HCSA from stored preds and gts
            from collections import defaultdict

            # Need sequences of dev set to recompute. Can't access here; skip recompute and instead
            # fetch last stored HCSA from metrics list
            hcs_val = lr_dict["metrics"]["val"][lr][-1][-1]
            final_hcs.append(hcs_val)

        plt.figure()
        plt.bar([str(lr) for lr in lrs], final_hcs)
        plt.xlabel("Learning Rate")
        plt.ylabel("Final Dev HCSA")
        plt.title("SPR_BENCH Final Dev HCSA for each Learning Rate")
        fname = os.path.join(working_dir, "SPR_BENCH_final_dev_HCSA_all_lr.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()
