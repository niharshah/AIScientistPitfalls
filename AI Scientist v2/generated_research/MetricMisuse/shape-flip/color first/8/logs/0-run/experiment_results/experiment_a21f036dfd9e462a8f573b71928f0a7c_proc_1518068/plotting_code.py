import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load -------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------- iterate datasets --------
for abl_name, ds_dict in experiment_data.items():
    for ds_name, rec in ds_dict.items():
        losses = rec.get("losses", {})
        metrics = rec.get("metrics", {})
        preds = np.asarray(rec.get("predictions", []))
        gts = np.asarray(rec.get("ground_truth", []))

        # ---------- loss curves ----------
        try:
            plt.figure()
            plt.plot(losses.get("train", []), label="Train")
            plt.plot(losses.get("val", []), label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds_name} Loss Curves\nTrain vs Val")
            plt.legend()
            fname = f"{ds_name}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
        except Exception as e:
            print(f"Error creating loss curve for {ds_name}: {e}")
        finally:
            plt.close()

        # ---------- accuracy curves ----------
        try:
            plt.figure()
            plt.plot(metrics.get("train", []), label="Train")
            plt.plot(metrics.get("val", []), label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{ds_name} Accuracy Curves\nTrain vs Val")
            plt.legend()
            fname = f"{ds_name}_accuracy_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
        except Exception as e:
            print(f"Error creating accuracy curve for {ds_name}: {e}")
        finally:
            plt.close()

        # ---------- confusion matrix ----------
        try:
            if preds.size and gts.size:
                num_cls = int(max(preds.max(), gts.max()) + 1)
                cm = np.zeros((num_cls, num_cls), dtype=int)
                for t, p in zip(gts, preds):
                    cm[t, p] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im, shrink=0.75)
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.title(f"{ds_name} Confusion Matrix\nCounts per class")
                for i in range(num_cls):
                    for j in range(num_cls):
                        plt.text(
                            j,
                            i,
                            cm[i, j],
                            ha="center",
                            va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontsize=8,
                        )
                fname = f"{ds_name}_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname))
        except Exception as e:
            print(f"Error creating confusion matrix for {ds_name}: {e}")
        finally:
            plt.close()
