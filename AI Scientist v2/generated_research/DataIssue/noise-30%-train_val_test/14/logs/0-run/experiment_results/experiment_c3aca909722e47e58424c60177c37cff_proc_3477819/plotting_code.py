import matplotlib.pyplot as plt
import numpy as np
import os
import itertools

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to create confusion matrix
def confusion_matrix(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    idx = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[idx[t], idx[p]] += 1
    return mat, labels


# iterate datasets / models
for dset_name, models in experiment_data.items():
    for model_name, entry in models.items():
        # ------- figure 1: loss curve -------
        try:
            epochs = [e["epoch"] for e in entry["losses"]["train"]]
            tr_loss = [e["loss"] for e in entry["losses"]["train"]]
            val_loss = [e["loss"] for e in entry["losses"]["val"]]
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.title(f"{dset_name} - {model_name}\nLoss vs Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            fname = f"{dset_name}_{model_name}_loss_curve.png".replace(" ", "_")
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {model_name}: {e}")
            plt.close()

        # ------- figure 2: macro-F1 curve -------
        try:
            tr_f1 = [e["macro_f1"] for e in entry["metrics"]["train"]]
            val_f1 = [e["macro_f1"] for e in entry["metrics"]["val"]]
            plt.figure()
            plt.plot(epochs, tr_f1, label="Train")
            plt.plot(epochs, val_f1, label="Validation")
            plt.title(f"{dset_name} - {model_name}\nMacro-F1 vs Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.legend()
            fname = f"{dset_name}_{model_name}_macroF1_curve.png".replace(" ", "_")
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating F1 plot for {model_name}: {e}")
            plt.close()

        # ------- figure 3: RGA / accuracy curve (validation) -------
        try:
            val_acc = [e["RGA"] for e in entry["metrics"]["val"]]
            if any(v is not None for v in val_acc):
                plt.figure()
                plt.plot(epochs, val_acc, marker="o")
                plt.title(f"{dset_name} - {model_name}\nRGA (Accuracy) vs Epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                fname = f"{dset_name}_{model_name}_accuracy_curve.png".replace(" ", "_")
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            else:
                print(f"No RGA data for {model_name}")
        except Exception as e:
            print(f"Error creating accuracy plot for {model_name}: {e}")
            plt.close()

        # ------- figure 4: confusion matrix -------
        try:
            y_pred = entry["predictions"]
            y_true = entry["ground_truth"]
            if len(y_pred) and len(y_true):
                cm, labels = confusion_matrix(y_true, y_pred)
                plt.figure()
                plt.imshow(cm, interpolation="nearest", cmap="Blues")
                plt.title(f"{dset_name} - {model_name}\nConfusion Matrix")
                plt.colorbar()
                tick_marks = np.arange(len(labels))
                plt.xticks(tick_marks, labels, rotation=45, ha="right")
                plt.yticks(tick_marks, labels)
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                fname = f"{dset_name}_{model_name}_confusion_matrix.png".replace(
                    " ", "_"
                )
                plt.tight_layout()
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {model_name}: {e}")
            plt.close()
