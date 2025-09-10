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

for abl_name, datasets in experiment_data.items():
    for dset_name, rec in datasets.items():
        # ---------- gather per-epoch data ----------
        train_losses = rec["losses"].get("train", [])
        val_losses = rec["losses"].get("val", [])
        epochs = np.arange(1, len(train_losses) + 1)

        # -------- 1) loss curves ----------
        try:
            plt.figure()
            plt.plot(epochs, train_losses, label="Train")
            plt.plot(epochs, val_losses, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset_name} Loss Curves\nAblation: {abl_name}")
            plt.legend()
            fname = f"{dset_name}_loss_curves_{abl_name}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curves for {dset_name}-{abl_name}: {e}")
            plt.close()

        # -------- 2) metric curves ----------
        try:
            swa = [m.get("swa") for m in rec["metrics"].get("val", [])]
            cwa = [m.get("cwa") for m in rec["metrics"].get("val", [])]
            ccwa = [m.get("ccwa") for m in rec["metrics"].get("val", [])]
            if any(v is not None for v in ccwa):
                plt.figure()
                plt.plot(epochs, swa, label="SWA")
                plt.plot(epochs, cwa, label="CWA")
                plt.plot(epochs, ccwa, label="CCWA")
                plt.xlabel("Epoch")
                plt.ylabel("Score")
                plt.title(f"{dset_name} Weighted Accuracies\nAblation: {abl_name}")
                plt.legend()
                fname = f"{dset_name}_metrics_{abl_name}.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating metric curves for {dset_name}-{abl_name}: {e}")
            plt.close()

        # -------- 3) confusion matrix ----------
        try:
            preds = rec.get("predictions", [])
            gts = rec.get("ground_truth", [])
            if preds and gts:
                labels = sorted(set(gts) | set(preds))
                cm = np.zeros((len(labels), len(labels)), dtype=int)
                for t, p in zip(gts, preds):
                    cm[t, p] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"{dset_name} Confusion Matrix\nAblation: {abl_name}")
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        plt.text(
                            j, i, cm[i, j], ha="center", va="center", color="black"
                        )
                fname = f"{dset_name}_conf_matrix_{abl_name}.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {dset_name}-{abl_name}: {e}")
            plt.close()
