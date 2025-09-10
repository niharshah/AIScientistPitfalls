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

# store final val accuracies for optional comparison
val_acc_summary = {}

for ds_name, ds in experiment_data.items():
    losses = ds.get("losses", {})
    metrics = ds.get("metrics", {})
    preds = np.asarray(ds.get("predictions", []))
    gts = np.asarray(ds.get("ground_truth", []))

    # --------- plot 1: loss curves ---------
    try:
        plt.figure()
        if losses.get("train"):
            plt.plot(losses["train"], label="Train")
        if losses.get("val"):
            plt.plot(losses["val"], label="Validation")
        plt.title(f"{ds_name} Loss Curve\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {ds_name}: {e}")
        plt.close()

    # --------- plot 2: accuracy curves ---------
    try:
        plt.figure()
        if metrics.get("train_acc"):
            plt.plot(metrics["train_acc"], label="Train")
        if metrics.get("val_acc"):
            plt.plot(metrics["val_acc"], label="Validation")
        plt.title(f"{ds_name} Accuracy Curve\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve for {ds_name}: {e}")
        plt.close()

    # --------- plot 3: CXA curves ---------
    try:
        if metrics.get("train_cxa") or metrics.get("val_cxa"):
            plt.figure()
            if metrics.get("train_cxa"):
                plt.plot(metrics["train_cxa"], label="Train CXA")
            if metrics.get("val_cxa"):
                plt.plot(metrics["val_cxa"], label="Validation CXA")
            plt.title(f"{ds_name} CXA Curve\nLeft: Train, Right: Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Complexity-weighted Accuracy")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_CXA_curve.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating CXA curve for {ds_name}: {e}")
        plt.close()

    # --------- plot 4: confusion matrix ---------
    try:
        if preds.size and gts.size:
            num_c = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((num_c, num_c), int)
            for g, p in zip(gts, preds):
                cm[g, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.title(
                f"{ds_name} Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            ticks = np.arange(num_c)
            plt.xticks(ticks, ticks)
            plt.yticks(ticks, ticks)
            plt.savefig(os.path.join(working_dir, f"{ds_name}_conf_matrix.png"))
            plt.close()
        else:
            print(
                f"Skipping confusion matrix for {ds_name}: empty predictions or labels."
            )
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()

    # record final val accuracy for comparison
    if metrics.get("val_acc"):
        val_acc_summary[ds_name] = metrics["val_acc"][-1]

# --------- comparison bar plot across datasets ---------
try:
    if len(val_acc_summary) > 1:
        plt.figure()
        names, vals = zip(*val_acc_summary.items())
        plt.bar(names, vals)
        plt.ylabel("Final Validation Accuracy")
        plt.title("Dataset Comparison of Final Validation Accuracy")
        plt.savefig(os.path.join(working_dir, "datasets_val_accuracy_comparison.png"))
        plt.close()
except Exception as e:
    print(f"Error creating dataset comparison plot: {e}")
    plt.close()
