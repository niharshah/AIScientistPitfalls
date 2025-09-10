import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- paths & load -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ----------------- iterate experiments ----------
for exp_name, datasets in experiment_data.items():
    for dset, data in datasets.items():
        epochs = np.arange(1, len(data["losses"]["train"]) + 1)

        # ---------- loss curve -------------
        try:
            plt.figure()
            plt.plot(epochs, data["losses"]["train"], label="Train")
            plt.plot(epochs, data["losses"]["val"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{exp_name} – {dset}\nLoss Curve")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_{exp_name}_loss_curve.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {dset}: {e}")
            plt.close()

        # ---------- metric curve -----------
        try:
            plt.figure()
            plt.plot(epochs, data["metrics"]["train"], label="Train")
            plt.plot(epochs, data["metrics"]["val"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Comp-Weighted Accuracy")
            plt.title(f"{exp_name} – {dset}\nCWA Curve")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_{exp_name}_cwa_curve.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating metric curve for {dset}: {e}")
            plt.close()

        # ---------- confusion matrix -------
        try:
            if data.get("predictions") and data.get("ground_truth"):
                preds = np.array(data["predictions"])
                gts = np.array(data["ground_truth"])
                n_cls = max(preds.max(), gts.max()) + 1
                cm = np.zeros((n_cls, n_cls), dtype=int)
                for t, p in zip(gts, preds):
                    cm[t, p] += 1
                plt.figure()
                plt.imshow(cm, cmap="Blues")
                plt.colorbar()
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title(f"{exp_name} – {dset}\nConfusion Matrix (best epoch)")
                fname = os.path.join(working_dir, f"{dset}_{exp_name}_conf_matrix.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {dset}: {e}")
            plt.close()

        # ---------- print final metric -----
        if data["metrics"]["val"]:
            print(f"{exp_name}/{dset} final Val CWA: {data['metrics']['val'][-1]:.4f}")
