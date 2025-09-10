import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------
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

# iterate over model/dataset combinations
for model_name, dsets in experiment_data.items():
    for dset_name, content in dsets.items():
        losses = content.get("losses", {})
        metrics = content.get("metrics", {})
        preds = content.get("predictions", [])
        gts = content.get("ground_truth", [])

        # 1) train/val loss curve -------------------------------------------------
        try:
            train_loss = losses.get("train", [])
            val_loss = losses.get("val", [])
            if train_loss and val_loss:
                plt.figure()
                epochs = range(1, len(train_loss) + 1)
                plt.plot(epochs, train_loss, label="Train")
                plt.plot(epochs, val_loss, label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                plt.title(f"{dset_name} - {model_name} - Loss Curve")
                plt.legend()
                fname = f"{dset_name}_{model_name}_loss_curve.png".replace(" ", "_")
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve: {e}")
            plt.close()

        # 2) train/val CWCA curve -------------------------------------------------
        try:
            train_cwca = metrics.get("train", [])
            val_cwca = metrics.get("val", [])
            if train_cwca and val_cwca:
                plt.figure()
                epochs = range(1, len(train_cwca) + 1)
                plt.plot(epochs, train_cwca, label="Train")
                plt.plot(epochs, val_cwca, label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("CWCA")
                plt.title(f"{dset_name} - {model_name} - CWCA Curve")
                plt.legend()
                fname = f"{dset_name}_{model_name}_cwca_curve.png".replace(" ", "_")
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating CWCA curve: {e}")
            plt.close()

        # 3) confusion matrix on test set ----------------------------------------
        try:
            if preds and gts:
                cm = confusion_matrix(gts, preds)
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xlabel("Predicted label")
                plt.ylabel("True label")
                plt.title(f"{dset_name} - {model_name} - Confusion Matrix (Test Set)")
                for (i, j), v in np.ndenumerate(cm):
                    plt.text(j, i, str(v), ha="center", va="center", color="black")
                fname = f"{dset_name}_{model_name}_confusion_matrix.png".replace(
                    " ", "_"
                )
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            plt.close()

        # print evaluation metric -------------------------------------------------
        try:
            test_cwca = metrics.get("test", [None])[0]
            print(f"{model_name} on {dset_name} - Test CWCA: {test_cwca}")
        except Exception as e:
            print(f"Error printing test CWCA: {e}")
