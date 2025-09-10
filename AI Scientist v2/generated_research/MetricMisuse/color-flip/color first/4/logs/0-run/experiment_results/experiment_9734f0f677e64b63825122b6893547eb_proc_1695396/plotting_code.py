import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data --------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# iterate through experiments --------------------------------------------------
for exp_name, datasets in experiment_data.items():
    for dset_name, rec in datasets.items():
        losses = rec.get("losses", {})
        metrics = rec.get("metrics", {}).get("val", [])
        preds = rec.get("predictions", [])
        gts = rec.get("ground_truth", [])

        # --- 1. loss curves ---------------------------------------------------
        try:
            tr_loss = losses.get("train", [])
            val_loss = losses.get("val", [])
            if tr_loss and val_loss:
                plt.figure()
                epochs = range(1, len(tr_loss) + 1)
                plt.plot(epochs, tr_loss, label="Train")
                plt.plot(epochs, val_loss, label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{dset_name} Loss Curves ({exp_name})")
                plt.legend()
                fname = f"{exp_name}_{dset_name}_loss_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve: {e}")
            plt.close()

        # --- 2. validation accuracy ------------------------------------------
        try:
            acc = [m["acc"] for m in metrics] if metrics else []
            if acc:
                plt.figure()
                plt.plot(range(1, len(acc) + 1), acc, marker="o")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.title(f"{dset_name} Validation Accuracy ({exp_name})")
                fname = f"{exp_name}_{dset_name}_val_accuracy.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot: {e}")
            plt.close()

        # --- 3. weighted accuracies ------------------------------------------
        try:
            if metrics:
                cwa = [m["CWA"] for m in metrics]
                swa = [m["SWA"] for m in metrics]
                comp = [m["CompWA"] for m in metrics]
                plt.figure()
                ep = range(1, len(metrics) + 1)
                plt.plot(ep, cwa, label="CWA")
                plt.plot(ep, swa, label="SWA")
                plt.plot(ep, comp, label="CompWA")
                plt.xlabel("Epoch")
                plt.ylabel("Score")
                plt.title(f"{dset_name} Weighted Accuracies ({exp_name})")
                plt.legend()
                fname = f"{exp_name}_{dset_name}_weighted_accuracies.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating weighted accuracy plot: {e}")
            plt.close()

        # --- 4. confusion matrix (final) -------------------------------------
        try:
            if preds and gts:
                n_cls = int(max(max(preds), max(gts))) + 1
                cm = np.zeros((n_cls, n_cls), dtype=int)
                for t, p in zip(gts, preds):
                    cm[t, p] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.title(f"{dset_name} Confusion Matrix ({exp_name})")
                for i in range(n_cls):
                    for j in range(n_cls):
                        plt.text(
                            j, i, cm[i, j], ha="center", va="center", color="black"
                        )
                fname = f"{exp_name}_{dset_name}_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            plt.close()
