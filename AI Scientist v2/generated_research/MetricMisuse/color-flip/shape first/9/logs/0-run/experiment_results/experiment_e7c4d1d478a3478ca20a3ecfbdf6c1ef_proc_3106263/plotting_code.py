import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- load data -------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------- plotting --------------
for model_name, datasets in experiment_data.items():
    for ds_name, ds_dict in datasets.items():
        losses = ds_dict.get("losses", {})
        metrics = ds_dict.get("metrics", {})
        preds = ds_dict.get("predictions", [])
        gts = ds_dict.get("ground_truth", [])

        # ---- 1: loss curves ----
        try:
            if losses.get("train") or losses.get("val"):
                plt.figure()
                if losses.get("train"):
                    plt.plot(losses["train"], label="Train Loss")
                if losses.get("val"):
                    plt.plot(losses["val"], label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{ds_name} Loss Curves\nLeft: Train, Right: Validation")
                plt.legend()
                fname = f"{model_name}_{ds_name}_loss_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {model_name}-{ds_name}: {e}")
            plt.close()

        # ---- 2: metric curves ----
        try:
            if metrics.get("train") or metrics.get("val"):
                plt.figure()
                if metrics.get("train"):
                    plt.plot(metrics["train"], label="Train Metric")
                if metrics.get("val"):
                    plt.plot(metrics["val"], label="Validation CompWA")
                plt.xlabel("Epoch")
                plt.ylabel("Comp-Weighted-Acc")
                plt.title(f"{ds_name} CompWA over Epochs\nValidation performance")
                plt.legend()
                fname = f"{model_name}_{ds_name}_metric_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating metric plot for {model_name}-{ds_name}: {e}")
            plt.close()

        # ---- 3: confusion matrix ----
        try:
            if len(preds) and len(gts):
                labels = sorted(set(gts) | set(preds))
                cm = np.zeros((len(labels), len(labels)), dtype=int)
                lab2idx = {l: i for i, l in enumerate(labels)}
                for g, p in zip(gts, preds):
                    cm[lab2idx[g], lab2idx[p]] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.xticks(range(len(labels)), labels, rotation=90)
                plt.yticks(range(len(labels)), labels)
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.title(f"{ds_name} Confusion Matrix\nLeft: GT, Right: Predicted")
                fname = f"{model_name}_{ds_name}_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname), bbox_inches="tight")
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {model_name}-{ds_name}: {e}")
            plt.close()
