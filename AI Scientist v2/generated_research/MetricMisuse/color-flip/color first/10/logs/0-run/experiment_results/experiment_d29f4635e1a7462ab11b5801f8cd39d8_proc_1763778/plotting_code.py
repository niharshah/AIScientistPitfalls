import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for model_name, dsets in experiment_data.items():
    for dset_name, bundle in dsets.items():
        metrics = bundle.get("metrics", {})
        preds = np.asarray(bundle.get("predictions", []))
        gts = np.asarray(bundle.get("ground_truth", []))
        epochs = np.arange(1, len(metrics.get("train_loss", [])) + 1)

        # 1) Train / Val loss curve
        try:
            plt.figure()
            plt.plot(
                epochs, metrics.get("train_loss", []), marker="o", label="Train Loss"
            )
            plt.plot(
                epochs, metrics.get("val_loss", []), marker="s", label="Validation Loss"
            )
            plt.xlabel("Epoch")
            plt.ylabel("BCE Loss")
            plt.title(f"{model_name} on {dset_name}\nTraining vs Validation Loss")
            plt.legend()
            fname = os.path.join(
                working_dir, f"{dset_name}_{model_name}_loss_curve.png"
            )
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {model_name}-{dset_name}: {e}")
            plt.close()

        # 2) Weighted accuracy curves
        try:
            plt.figure()
            for k, lab in [
                ("val_CWA", "CWA"),
                ("val_SWA", "SWA"),
                ("val_CWA2", "CWA2"),
            ]:
                if k in metrics:
                    plt.plot(epochs, metrics[k], marker="o", label=lab)
            plt.xlabel("Epoch")
            plt.ylabel("Weighted Accuracy")
            plt.title(f"{model_name} on {dset_name}\nCWA / SWA / CWA2 over Epochs")
            plt.legend()
            fname = os.path.join(
                working_dir, f"{dset_name}_{model_name}_weighted_acc.png"
            )
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(
                f"Error creating weighted accuracy plot for {model_name}-{dset_name}: {e}"
            )
            plt.close()

        # 3) Confusion matrix heat-map from final predictions
        try:
            if preds.size and gts.size and preds.shape == gts.shape:
                TP = int(((preds == 1) & (gts == 1)).sum())
                TN = int(((preds == 0) & (gts == 0)).sum())
                FP = int(((preds == 1) & (gts == 0)).sum())
                FN = int(((preds == 0) & (gts == 1)).sum())
                cm = np.array([[TP, FP], [FN, TN]])
                plt.figure()
                plt.imshow(cm, cmap="Blues")
                for i in range(2):
                    for j in range(2):
                        plt.text(
                            j, i, cm[i, j], ha="center", va="center", color="black"
                        )
                plt.xticks([0, 1], ["Pred 1", "Pred 0"])
                plt.yticks([0, 1], ["True 1", "True 0"])
                plt.colorbar()
                plt.title(f"{model_name} on {dset_name}\nConfusion Matrix (Dev Set)")
                fname = os.path.join(
                    working_dir, f"{dset_name}_{model_name}_confusion_matrix.png"
                )
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {model_name}-{dset_name}: {e}")
            plt.close()
