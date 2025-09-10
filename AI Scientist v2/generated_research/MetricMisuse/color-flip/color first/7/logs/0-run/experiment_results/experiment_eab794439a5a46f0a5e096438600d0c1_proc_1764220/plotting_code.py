import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------
for exp_name, datasets in experiment_data.items():
    for dset_name, info in datasets.items():
        epochs = info.get("epochs", [])
        losses = info.get("losses", {})
        metrics = info.get("metrics", {})
        preds = info.get("predictions", [])
        gts = info.get("ground_truth", [])

        # ------------------- 1. Loss curves -------------------------
        try:
            if epochs and losses:
                plt.figure()
                if "train" in losses:
                    plt.plot(epochs, losses["train"], label="train")
                if "val" in losses:
                    plt.plot(epochs, losses["val"], label="val")
                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                plt.title(f"{dset_name} – Train/Val Loss Curves")
                plt.legend()
                fname = f"{dset_name}_loss_curves.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curves for {dset_name}: {e}")
            plt.close()

        # ------------------- 2. Metric curves -----------------------
        try:
            if epochs and metrics:
                plt.figure()
                for split in ["train", "val"]:
                    if split in metrics:
                        cwa = [m["cwa"] for m in metrics[split]]
                        swa = [m["swa"] for m in metrics[split]]
                        cpx = [m["cpx"] for m in metrics[split]]
                        plt.plot(
                            epochs,
                            cwa,
                            label=f"{split}_CWA",
                            linestyle="--" if split == "val" else "-",
                        )
                        plt.plot(
                            epochs,
                            swa,
                            label=f"{split}_SWA",
                            linestyle="--" if split == "val" else "-",
                        )
                        plt.plot(
                            epochs,
                            cpx,
                            label=f"{split}_CPX",
                            linestyle="--" if split == "val" else "-",
                        )
                plt.xlabel("Epoch")
                plt.ylabel("Weighted Accuracy")
                plt.title(f"{dset_name} – Weighted Accuracy Metrics")
                plt.legend()
                fname = f"{dset_name}_metric_curves.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating metric curves for {dset_name}: {e}")
            plt.close()

        # ------------------- 3. Confusion matrix --------------------
        try:
            if preds and gts:
                import itertools

                labels = sorted(set(gts) | set(preds))
                cm = np.zeros((len(labels), len(labels)), dtype=int)
                for t, p in zip(gts, preds):
                    cm[labels.index(t), labels.index(p)] += 1
                plt.figure()
                plt.imshow(cm, cmap="Blues")
                plt.colorbar()
                plt.xticks(range(len(labels)), labels)
                plt.yticks(range(len(labels)), labels)
                for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
                plt.title(
                    f"{dset_name} – Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
                )
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                fname = f"{dset_name}_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {dset_name}: {e}")
            plt.close()
