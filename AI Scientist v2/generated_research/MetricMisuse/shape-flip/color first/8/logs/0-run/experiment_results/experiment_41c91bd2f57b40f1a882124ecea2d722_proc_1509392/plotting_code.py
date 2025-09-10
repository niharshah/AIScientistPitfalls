import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- working dir ----------
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

# ---------- iterate over datasets ----------
for ds_name, ds_content in experiment_data.items():
    losses = ds_content.get("losses", {})
    metrics = ds_content.get("metrics", {})
    preds = np.array(ds_content.get("predictions", []))
    gts = np.array(ds_content.get("ground_truth", []))

    # --------- plot 1: loss curves ---------
    try:
        plt.figure()
        if "train" in losses and losses["train"]:
            plt.plot(losses["train"], label="Train")
        if "val" in losses and losses["val"]:
            plt.plot(losses["val"], label="Validation")
        plt.title(f"{ds_name} Loss Curve\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"{ds_name}_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {ds_name}: {e}")
        plt.close()

    # --------- plot 2: accuracy curves ---------
    try:
        plt.figure()
        if "train" in metrics and metrics["train"]:
            plt.plot(metrics["train"], label="Train")
        if "val" in metrics and metrics["val"]:
            plt.plot(metrics["val"], label="Validation")
        plt.title(f"{ds_name} Accuracy Curve\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = f"{ds_name}_accuracy_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve for {ds_name}: {e}")
        plt.close()

    # --------- plot 3: confusion matrix ---------
    try:
        if preds.size and gts.size:
            num_classes = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for g, p in zip(gts, preds):
                cm[g, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f"{ds_name} Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            ticks = np.arange(num_classes)
            plt.xticks(ticks, [f"c{i}" for i in ticks])
            plt.yticks(ticks, [f"c{i}" for i in ticks])
            fname = f"{ds_name}_confusion_matrix.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        else:
            print(
                f"Skipping confusion matrix for {ds_name}: empty predictions or ground truth."
            )
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()

    # --------- print summary metric ----------
    if metrics.get("val"):
        print(f'{ds_name} final validation accuracy: {metrics["val"][-1]:.4f}')
