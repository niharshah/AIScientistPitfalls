import matplotlib.pyplot as plt
import numpy as np
import os

# --- setup ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- load experiment data ---
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# --- iterate over entries ---
for model_name, datasets in experiment_data.items():
    for dset_name, record in datasets.items():
        # helpers
        ep_labels = list(range(1, len(record["losses"]["train"]) + 1))
        # 1) Loss curves -------------------------------------------------------
        try:
            plt.figure()
            plt.plot(ep_labels, record["losses"]["train"], label="train")
            plt.plot(ep_labels, record["losses"]["val"], label="val")
            plt.xlabel("Epoch")
            plt.ylabel("BCE Loss")
            plt.title(f"{dset_name} – {model_name}\nTraining vs Validation Loss")
            plt.legend()
            fname = f"{dset_name}_{model_name}_loss_curve.png"
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dset_name}-{model_name}: {e}")
            plt.close()
        # 2) MCC curves --------------------------------------------------------
        try:
            plt.figure()
            plt.plot(ep_labels, record["metrics"]["train_MCC"], label="train_MCC")
            plt.plot(ep_labels, record["metrics"]["val_MCC"], label="val_MCC")
            plt.xlabel("Epoch")
            plt.ylabel("MCC")
            plt.title(f"{dset_name} – {model_name}\nTraining vs Validation MCC")
            plt.legend()
            fname = f"{dset_name}_{model_name}_mcc_curve.png"
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating MCC plot for {dset_name}-{model_name}: {e}")
            plt.close()
        # 3) Confusion matrix style bar plot ----------------------------------
        try:
            preds = np.array(record.get("predictions", []))
            gts = np.array(record.get("ground_truth", []))
            if preds.size and gts.size:
                tp = np.sum((preds == 1) & (gts == 1))
                fp = np.sum((preds == 1) & (gts == 0))
                tn = np.sum((preds == 0) & (gts == 0))
                fn = np.sum((preds == 0) & (gts == 1))
                plt.figure()
                plt.bar(
                    ["TP", "FP", "TN", "FN"],
                    [tp, fp, tn, fn],
                    color=["g", "r", "b", "y"],
                )
                plt.ylabel("Count")
                plt.title(f"{dset_name} – {model_name}\nConfusion Matrix Counts (Test)")
                fname = f"{dset_name}_{model_name}_confusion_counts.png"
                plt.tight_layout()
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating confusion plot for {dset_name}-{model_name}: {e}")
            plt.close()
        # 4) Label distribution -----------------------------------------------
        try:
            if preds.size and gts.size:
                plt.figure()
                plt.hist(
                    [gts, preds],
                    bins=[-0.5, 0.5, 1.5],
                    label=["Ground Truth", "Predictions"],
                    alpha=0.7,
                )
                plt.xticks([0, 1])
                plt.xlabel("Label")
                plt.ylabel("Count")
                plt.title(f"{dset_name} – {model_name}\nLabel Distribution (Test)")
                plt.legend()
                fname = f"{dset_name}_{model_name}_label_distribution.png"
                plt.tight_layout()
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(
                f"Error creating label distribution for {dset_name}-{model_name}: {e}"
            )
            plt.close()
        # 5) Print stored metrics ---------------------------------------------
        print(
            f"{dset_name} – {model_name}: Test MCC={record.get('test_MCC', 'NA'):.3f}, "
            f"Test macro-F1={record.get('test_F1', 'NA'):.3f}"
        )
