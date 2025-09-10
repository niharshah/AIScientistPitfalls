import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- paths & loading -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    for dset, rec in experiment_data.items():
        # -------- Figure 1: Loss curves --------
        try:
            tr_losses = rec["losses"].get("train")
            val_losses = rec["losses"].get("val")
            if tr_losses is not None and val_losses is not None and len(tr_losses):
                plt.figure()
                plt.plot(tr_losses, "--", label="train")
                plt.plot(val_losses, "-", label="validation")
                plt.title(f"{dset} Loss Curves\n(Training vs Validation)")
                plt.xlabel("Epoch")
                plt.ylabel("Total Loss")
                plt.legend()
                plt.tight_layout()
                save_path = os.path.join(working_dir, f"{dset}_loss_curves.png")
                plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dset}: {e}")
            plt.close()

        # -------- Figure 2: Validation SWA --------
        try:
            val_metrics = rec["metrics"].get("val")
            if val_metrics is not None and len(val_metrics):
                plt.figure()
                plt.plot(val_metrics, marker="o")
                plt.title(f"{dset} Validation Shape-Weighted Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("SWA")
                plt.tight_layout()
                save_path = os.path.join(working_dir, f"{dset}_val_SWA.png")
                plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error creating SWA plot for {dset}: {e}")
            plt.close()

        # -------- Figure 3: Confusion matrix --------
        try:
            preds = rec.get("predictions")
            gts = rec.get("ground_truth")
            if preds is not None and gts is not None and len(preds):
                labels = sorted(set(gts))
                idx = {l: i for i, l in enumerate(labels)}
                cm = np.zeros((len(labels), len(labels)), dtype=int)
                for t, p in zip(gts, preds):
                    cm[idx[t], idx[p]] += 1

                plt.figure(figsize=(6, 5))
                plt.imshow(cm, cmap="Blues")
                plt.colorbar()
                plt.title(f"{dset} Confusion Matrix\n(Test Set Predictions)")
                plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
                plt.yticks(range(len(labels)), labels, fontsize=6)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                # annotate
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        txt_color = "white" if cm[i, j] > cm.max() / 2 else "black"
                        plt.text(
                            j,
                            i,
                            cm[i, j],
                            ha="center",
                            va="center",
                            color=txt_color,
                            fontsize=6,
                        )
                plt.tight_layout()
                save_path = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
                plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {dset}: {e}")
            plt.close()
