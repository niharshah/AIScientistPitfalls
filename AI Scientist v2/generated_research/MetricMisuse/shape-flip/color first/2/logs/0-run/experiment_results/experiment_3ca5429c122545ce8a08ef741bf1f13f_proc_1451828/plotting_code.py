import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- SETUP ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- LOAD DATA ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to compute confusion matrix & accuracy
def confusion_and_acc(y_true, y_pred, num_classes=2):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = np.trace(cm) / np.sum(cm) if cm.sum() else 0.0
    return cm, acc


# ---------- PLOTTING ----------
for abl_key, dsets in experiment_data.items():
    for dataset_name, entry in dsets.items():
        # 1) Loss curves ----------------------------------------------------
        try:
            tr = np.array(entry["losses"]["train"])
            val = np.array(entry["losses"]["val"])
            if tr.size and val.size:
                plt.figure()
                plt.plot(tr[:, 0], tr[:, 1], label="Train")
                plt.plot(val[:, 0], val[:, 1], label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{dataset_name} Loss Curve\nAblation: {abl_key}")
                plt.legend()
                fname = f"{dataset_name}_loss_curve_{abl_key}.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {dataset_name}: {e}")
            plt.close()

        # 2) PCWA curves ----------------------------------------------------
        try:
            tr = np.array(entry["metrics"]["train"])
            val = np.array(entry["metrics"]["val"])
            if tr.size and val.size:
                plt.figure()
                plt.plot(tr[:, 0], tr[:, 1], label="Train")
                plt.plot(val[:, 0], val[:, 1], label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("PCWA")
                plt.title(f"{dataset_name} PCWA Curve\nAblation: {abl_key}")
                plt.legend()
                fname = f"{dataset_name}_pcwa_curve_{abl_key}.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating PCWA curve for {dataset_name}: {e}")
            plt.close()

        # 3) Confusion matrix on test set -----------------------------------
        try:
            y_true = np.array(entry.get("ground_truth", []))
            y_pred = np.array(entry.get("predictions", []))
            if y_true.size and y_pred.size:
                cm, acc = confusion_and_acc(y_true, y_pred, num_classes=2)
                plt.figure()
                plt.imshow(cm, cmap="Blues")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(
                            j, i, cm[i, j], ha="center", va="center", color="black"
                        )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(
                    f"{dataset_name} Confusion Matrix (ACC={acc:.3f})\nAblation: {abl_key}"
                )
                plt.colorbar()
                fname = f"{dataset_name}_conf_mat_{abl_key}.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
                print(f"{dataset_name} Test Accuracy ({abl_key}): {acc:.4f}")
        except Exception as e:
            print(f"Error creating confusion matrix for {dataset_name}: {e}")
            plt.close()
