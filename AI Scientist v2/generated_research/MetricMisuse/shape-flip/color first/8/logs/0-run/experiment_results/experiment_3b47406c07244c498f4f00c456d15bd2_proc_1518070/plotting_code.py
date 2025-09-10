import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data -------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------- iterate & plot -------------------
for setting, dsets in experiment_data.items():
    for dname, ddict in dsets.items():
        # ------- helpers -------
        def _subsample(arr, max_len=5):
            if len(arr) <= max_len:
                return list(range(len(arr))), arr
            idx = np.linspace(0, len(arr) - 1, max_len, dtype=int)
            return idx, [arr[i] for i in idx]

        # ---------------- loss curve ----------------
        try:
            tr_loss = ddict["losses"]["train"]
            val_loss = ddict["losses"]["val"]
            idx, tr_plot = _subsample(tr_loss)
            _, val_plot = _subsample(val_loss)
            plt.figure()
            plt.plot(idx, tr_plot, label="Train")
            plt.plot(idx, val_plot, label="Validation")
            plt.title(f"{dname}: Cross-Entropy Loss\nSetting: {setting}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fpath = os.path.join(working_dir, f"{dname}_loss_curve.png")
            plt.savefig(fpath)
            print("Saved", fpath)
        except Exception as e:
            print(f"Error creating loss curve for {dname}: {e}")
        finally:
            plt.close()

        # ---------------- metric curve ----------------
        try:
            tr_acc = ddict["metrics"]["train"]
            val_acc = ddict["metrics"]["val"]
            idx, tr_plot = _subsample(tr_acc)
            _, val_plot = _subsample(val_acc)
            plt.figure()
            plt.plot(idx, tr_plot, label="Train")
            plt.plot(idx, val_plot, label="Validation")
            plt.title(f"{dname}: Accuracy\nSetting: {setting}")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fpath = os.path.join(working_dir, f"{dname}_accuracy_curve.png")
            plt.savefig(fpath)
            print("Saved", fpath)
        except Exception as e:
            print(f"Error creating accuracy curve for {dname}: {e}")
        finally:
            plt.close()

        # ---------------- confusion matrix ----------------
        try:
            preds = ddict.get("predictions", [])
            gts = ddict.get("ground_truth", [])
            if preds and gts and len(preds) == len(gts):
                classes = sorted(set(gts + preds))
                cm = np.zeros((len(classes), len(classes)), dtype=int)
                for p, g in zip(preds, gts):
                    cm[g, p] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.title(f"{dname}: Confusion Matrix\nSetting: {setting}")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.xticks(classes)
                plt.yticks(classes)
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        plt.text(
                            j, i, cm[i, j], ha="center", va="center", color="black"
                        )
                fpath = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
                plt.savefig(fpath)
                print("Saved", fpath)
        except Exception as e:
            print(f"Error creating confusion matrix for {dname}: {e}")
        finally:
            plt.close()
