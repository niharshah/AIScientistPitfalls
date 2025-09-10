import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- plotting ----------
for exp_name, dsets in experiment_data.items():
    for ds_name, log in dsets.items():
        epochs = log.get("epochs", [])
        losses = log.get("losses", {})
        metrics = log.get("metrics", {})
        preds = np.asarray(log.get("predictions", []))
        gts = np.asarray(log.get("ground_truth", []))

        # ---- 1. loss curve ----
        try:
            plt.figure()
            if "train" in losses:
                plt.plot(epochs, losses["train"], label="train")
            if "dev" in losses:
                plt.plot(epochs, losses["dev"], label="dev")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name} Loss Curve ({exp_name})")
            plt.legend()
            fname = f"{ds_name}_{exp_name}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {ds_name}: {e}")
            plt.close()

        # ---- 2. PHA curve ----
        try:
            plt.figure()
            if "train_PHA" in metrics:
                plt.plot(epochs, metrics["train_PHA"], label="train_PHA")
            if "dev_PHA" in metrics:
                plt.plot(epochs, metrics["dev_PHA"], label="dev_PHA")
            plt.xlabel("Epoch")
            plt.ylabel("PHA")
            plt.title(f"{ds_name} PHA Curve ({exp_name})")
            plt.legend()
            fname = f"{ds_name}_{exp_name}_PHA_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating PHA curve for {ds_name}: {e}")
            plt.close()

        # ---- 3. confusion matrix ----
        try:
            if preds.size and gts.size:
                n_cls = int(max(preds.max(), gts.max())) + 1
                cm = np.zeros((n_cls, n_cls), dtype=int)
                for p, g in zip(preds, gts):
                    cm[g, p] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.title(f"{ds_name} Confusion Matrix ({exp_name})")
                fname = f"{ds_name}_{exp_name}_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {ds_name}: {e}")
            plt.close()
