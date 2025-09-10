import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------
for model_name, dsets in experiment_data.items():
    for dset_name, rec in dsets.items():
        # ------------------------------------------------------------------
        # 1) LOSS CURVES
        try:
            plt.figure()
            epochs = range(1, len(rec["losses"]["train"]) + 1)
            plt.plot(epochs, rec["losses"]["train"], label="Train Loss")
            plt.plot(epochs, rec["losses"]["val"], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("BCE Loss")
            plt.title(f"{dset_name} Loss Curves ({model_name})")
            plt.legend()
            fname = f"{dset_name}_{model_name}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dset_name}: {e}")
            plt.close()

        # ------------------------------------------------------------------
        # 2) MCC CURVES
        try:
            plt.figure()
            epochs = range(1, len(rec["metrics"]["train"]) + 1)
            plt.plot(epochs, rec["metrics"]["train"], label="Train MCC")
            plt.plot(epochs, rec["metrics"]["val"], label="Val MCC")
            plt.xlabel("Epoch")
            plt.ylabel("Matthews Corrcoef")
            plt.title(f"{dset_name} MCC Curves ({model_name})")
            plt.legend()
            fname = f"{dset_name}_{model_name}_mcc_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating MCC plot for {dset_name}: {e}")
            plt.close()

        # ------------------------------------------------------------------
        # 3) CONFUSION-MATRIX STYLE BAR PLOT FOR BEST RUN (limit ≤5 plots)
        try:
            preds_runs = rec.get("predictions", [])
            gts_runs = rec.get("ground_truth", [])
            if preds_runs and gts_runs:
                # Select runs sorted by val MCC (same order as 'metrics' val)
                val_mccs = rec["metrics"]["val"]
                best_indices = np.argsort(val_mccs)[-5:]  # at most 5 plots
                for idx in best_indices:
                    preds = preds_runs[idx].astype(int)
                    gts = gts_runs[idx].astype(int)
                    tn, fp, fn, tp = confusion_matrix(gts, preds, labels=[0, 1]).ravel()
                    plt.figure()
                    plt.bar(
                        ["TP", "FP", "TN", "FN"],
                        [tp, fp, tn, fn],
                        color=["g", "r", "b", "k"],
                    )
                    plt.ylabel("Count")
                    plt.title(f"{dset_name} Confusion Counts (run {idx}, {model_name})")
                    fname = f"{dset_name}_{model_name}_confusion_run{idx}.png"
                    plt.savefig(os.path.join(working_dir, fname))
                    plt.close()
        except Exception as e:
            print(f"Error creating confusion plot for {dset_name}: {e}")
            plt.close()
