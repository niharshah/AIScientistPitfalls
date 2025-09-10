import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper ----------
def confusion_matrix(gt, pr, num_classes=2):
    mat = np.zeros((num_classes, num_classes), dtype=int)
    for g, p in zip(gt, pr):
        mat[g, p] += 1
    return mat


# ---------- plotting ----------
for variant, ds_dict in experiment_data.items():
    for ds_name, logs in ds_dict.items():
        losses_tr = np.asarray(logs["losses"]["train"])
        losses_va = np.asarray(logs["losses"]["val"])
        mcc_tr = np.asarray(logs["metrics"]["train"])
        mcc_va = np.asarray(logs["metrics"]["val"])
        preds = np.asarray(logs.get("predictions", []))
        gts = np.asarray(logs.get("ground_truth", []))

        # 1) loss curves -------------------------------------------------------
        try:
            plt.figure()
            epochs = np.arange(1, len(losses_tr) + 1)
            plt.plot(epochs, losses_tr, label="Train")
            plt.plot(epochs, losses_va, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"Loss Curves – {ds_name} ({variant})")
            plt.legend()
            fname = f"{ds_name}_{variant}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {ds_name}/{variant}: {e}")
            plt.close()

        # 2) MCC curves --------------------------------------------------------
        try:
            plt.figure()
            plt.plot(epochs, mcc_tr, label="Train")
            plt.plot(epochs, mcc_va, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Matthews Correlation Coefficient")
            plt.title(f"MCC Curves – {ds_name} ({variant})")
            plt.legend()
            fname = f"{ds_name}_{variant}_mcc_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating MCC curve for {ds_name}/{variant}: {e}")
            plt.close()

        # 3) confusion matrix --------------------------------------------------
        if preds.size and gts.size:
            try:
                cm = confusion_matrix(gts, preds, num_classes=2)
                plt.figure()
                plt.imshow(cm, cmap="Blues")
                for i in range(2):
                    for j in range(2):
                        plt.text(
                            j, i, cm[i, j], ha="center", va="center", color="black"
                        )
                plt.xticks([0, 1], ["Pred 0", "Pred 1"])
                plt.yticks([0, 1], ["True 0", "True 1"])
                plt.title(f"Confusion Matrix – {ds_name} ({variant})")
                plt.colorbar()
                fname = f"{ds_name}_{variant}_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating confusion matrix for {ds_name}/{variant}: {e}")
                plt.close()

        # ---------- print final metrics ----------
        if preds.size and gts.size:
            # recompute metrics here in case they weren’t saved
            tp = ((preds == 1) & (gts == 1)).sum()
            tn = ((preds == 0) & (gts == 0)).sum()
            fp = ((preds == 1) & (gts == 0)).sum()
            fn = ((preds == 0) & (gts == 1)).sum()
            mcc = (tp * tn - fp * fn) / np.sqrt(
                (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8
            )
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            print(f"{ds_name}/{variant} -> Test MCC={mcc:.4f} | Test F1={f1:.4f}")
