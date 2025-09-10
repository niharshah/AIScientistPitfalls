import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for exp_name, exp_dict in experiment_data.items():
    for dset, res in exp_dict.items():
        losses_tr = res.get("losses", {}).get("train", [])
        losses_val = res.get("losses", {}).get("val", [])
        metrics_val = res.get("metrics", {}).get("val", [])
        preds = res.get("predictions", np.array([]))
        gts = res.get("ground_truth", np.array([]))
        ws = res.get("weights", np.array([]))
        epochs = range(1, len(losses_tr) + 1)

        # 1) Loss curves
        try:
            plt.figure()
            plt.plot(epochs, losses_tr, label="Train")
            plt.plot(epochs, losses_val, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset} Loss Curves ({exp_name})")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_loss_curves_{exp_name}.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating loss plot: {e}")
            plt.close()

        # 2) Metric curves
        try:
            plt.figure()
            mf1 = [m.get("macro_f1", np.nan) for m in metrics_val]
            cwa = [m.get("cwa", np.nan) for m in metrics_val]
            plt.plot(epochs, mf1, label="Macro-F1")
            plt.plot(epochs, cwa, label="CWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.ylim(0, 1)
            plt.title(f"{dset} Validation Metrics ({exp_name})")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_metric_curves_{exp_name}.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating metric plot: {e}")
            plt.close()

        # 3) Confusion matrix
        try:
            if preds.size and gts.size:
                cm = confusion_matrix(gts, preds, labels=sorted(np.unique(gts)))
                cm_norm = cm / cm.sum(axis=1, keepdims=True)
                plt.figure()
                im = plt.imshow(cm_norm, cmap="Blues")
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"{dset} Confusion Matrix ({exp_name})")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(
                            j, i, f"{cm[i,j]}", ha="center", va="center", color="black"
                        )
                fname = os.path.join(working_dir, f"{dset}_conf_matrix_{exp_name}.png")
                plt.savefig(fname)
                plt.close()
                print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            plt.close()

        # 4) Weight distribution
        try:
            if ws.size:
                plt.figure()
                plt.hist(ws, bins=30, color="gray")
                plt.xlabel("Example Weight")
                plt.ylabel("Count")
                plt.title(f"{dset} Weight Distribution ({exp_name})")
                fname = os.path.join(working_dir, f"{dset}_weight_hist_{exp_name}.png")
                plt.savefig(fname)
                plt.close()
                print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating weight hist: {e}")
            plt.close()

        # 5) Correctness vs weight scatter
        try:
            if ws.size and preds.size:
                correct = (preds == gts).astype(int)
                plt.figure()
                plt.scatter(ws, correct, alpha=0.3, s=10)
                plt.yticks([0, 1], ["Wrong", "Correct"])
                plt.xlabel("Weight")
                plt.title(f"{dset} Correctness vs Weight ({exp_name})")
                fname = os.path.join(
                    working_dir, f"{dset}_weight_vs_correct_{exp_name}.png"
                )
                plt.savefig(fname)
                plt.close()
                print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating scatter plot: {e}")
            plt.close()
