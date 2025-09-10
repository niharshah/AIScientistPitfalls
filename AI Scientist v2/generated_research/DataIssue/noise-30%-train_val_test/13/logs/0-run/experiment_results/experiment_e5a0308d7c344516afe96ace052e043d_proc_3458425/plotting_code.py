import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# prepare working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dset_name, d in experiment_data.items():
    epochs = d.get("epochs", [])
    train_loss = d.get("losses", {}).get("train", [])
    val_loss = d.get("losses", {}).get("val", [])
    train_f1 = d.get("metrics", {}).get("train_macro_f1", [])
    val_f1 = d.get("metrics", {}).get("val_macro_f1", [])
    y_pred = np.array(d.get("predictions", []))
    y_true = np.array(d.get("ground_truth", []))

    # 1) Loss curve
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.title(f"{dset_name} Loss Curve\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset_name}_loss_curve.png"))
    except Exception as e:
        print(f"Error creating loss curve for {dset_name}: {e}")
    finally:
        plt.close()

    # 2) Macro-F1 curve
    try:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        plt.title(f"{dset_name} Macro-F1 Curve\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        plt.ylim(0, 1)
        plt.savefig(os.path.join(working_dir, f"{dset_name}_f1_curve.png"))
    except Exception as e:
        print(f"Error creating F1 curve for {dset_name}: {e}")
    finally:
        plt.close()

    # 3) Confusion matrix of final validation predictions
    if y_true.size and y_pred.size:
        try:
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(cmap="Blues", colorbar=False)
            plt.title(
                f"{dset_name} Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            plt.savefig(os.path.join(working_dir, f"{dset_name}_confusion_matrix.png"))
        except Exception as e:
            print(f"Error creating confusion matrix for {dset_name}: {e}")
        finally:
            plt.close()

    # print final metric
    if val_f1:
        print(f"{dset_name} final validation macro-F1: {val_f1[-1]:.4f}")
