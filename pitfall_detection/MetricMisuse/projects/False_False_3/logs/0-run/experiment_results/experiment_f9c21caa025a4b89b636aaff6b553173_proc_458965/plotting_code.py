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
    datasets = list(experiment_data.keys())
    test_swa_all = {}

    for dset in datasets:
        data = experiment_data[dset]

        # ---------- Figure 1: loss curves ----------
        try:
            plt.figure()
            if data["losses"]["train"]:
                plt.plot(data["losses"]["train"], "--", label="train")
            if data["losses"]["val"]:
                plt.plot(data["losses"]["val"], "-", label="val")
            plt.title(f"{dset} Training vs Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend(fontsize=6)
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset}_loss_curves.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dset}: {e}")
            plt.close()

        # ---------- Figure 2: validation SWA ----------
        try:
            plt.figure()
            if data["metrics"]["val"]:
                plt.plot(data["metrics"]["val"], label="SWA")
                plt.title(f"{dset} Validation Shape-Weighted Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("SWA")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(working_dir, f"{dset}_val_SWA.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating SWA plot for {dset}: {e}")
            plt.close()

        # ---------- Figure 3: confusion matrix ----------
        try:
            from itertools import product

            preds = data.get("predictions", [])
            gts = data.get("ground_truth", [])
            if preds and gts:
                labels = sorted(set(gts))
                idx = {l: i for i, l in enumerate(labels)}
                cm = np.zeros((len(labels), len(labels)), dtype=int)
                for t, p in zip(gts, preds):
                    cm[idx[t], idx[p]] += 1

                plt.figure()
                plt.imshow(cm, cmap="Blues")
                plt.colorbar()
                plt.title(f"{dset} Confusion Matrix")
                plt.xticks(range(len(labels)), labels, rotation=90)
                plt.yticks(range(len(labels)), labels)
                for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=6,
                    )
                plt.tight_layout()
                plt.savefig(os.path.join(working_dir, f"{dset}_confusion_matrix.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {dset}: {e}")
            plt.close()

        # collect final test SWA
        if "metrics" in data and "test" in data["metrics"]:
            test_swa_all[dset] = data["metrics"]["test"]

    # ---------- Figure 4: cross-dataset SWA comparison ----------
    try:
        if test_swa_all:
            plt.figure()
            names = list(test_swa_all.keys())
            swa_vals = [test_swa_all[n] for n in names]
            plt.bar(range(len(names)), swa_vals)
            plt.title("Test Shape-Weighted Accuracy Across Datasets")
            plt.ylabel("SWA")
            plt.xticks(range(len(names)), names, rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "datasets_test_SWA_comparison.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating cross-dataset comparison plot: {e}")
        plt.close()
