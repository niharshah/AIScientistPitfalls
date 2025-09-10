import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# House-keeping
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_key = "SPR_BENCH"
d = experiment_data.get(ds_key, {})
metrics = d.get("metrics", {})
losses = d.get("losses", {})
train_acc = metrics.get("train_acc", [])
val_acc = metrics.get("val_acc", [])
train_loss = losses.get("train", [])
val_loss = metrics.get("val_loss", [])
val_rfs = metrics.get("val_rfs", [])
preds = np.asarray(d.get("predictions", []))
gts = np.asarray(d.get("ground_truth", []))
rule_preds = np.asarray(d.get("rule_preds", []))
test_acc = d.get("test_acc", None)
test_rfs = d.get("test_rfs", None)

epochs = np.arange(1, len(train_acc) + 1)

# ------------------------------------------------------------------
# 1) Accuracy curves
# ------------------------------------------------------------------
try:
    if train_acc and val_acc:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_key}: Training vs Validation Accuracy")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{ds_key}_acc_curves.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Loss curves
# ------------------------------------------------------------------
try:
    if train_loss and val_loss:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.title(f"{ds_key}: Training vs Validation Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{ds_key}_loss_curves.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Rule-Fidelity curve
# ------------------------------------------------------------------
try:
    if val_rfs:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, val_rfs, color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("Validation RFS")
        plt.title(f"{ds_key}: Rule Fidelity Over Epochs")
        plt.ylim(0, 1)
        plt.savefig(
            os.path.join(working_dir, f"{ds_key}_val_rfs_curve.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
except Exception as e:
    print(f"Error creating RFS curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 4) Confusion matrix
# ------------------------------------------------------------------
try:
    if preds.size and gts.size:
        classes = sorted(np.unique(np.concatenate([gts, preds])))
        cm = np.zeros((len(classes), len(classes)), int)
        for gt, pr in zip(gts, preds):
            cm[gt, pr] += 1
        plt.figure(figsize=(4, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046)
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"{ds_key}: Confusion Matrix (Test)")
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        plt.savefig(
            os.path.join(working_dir, f"{ds_key}_confusion_matrix.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------------------------------------------------------
# 5) Test accuracy vs Rule fidelity bar chart
# ------------------------------------------------------------------
try:
    if test_acc is not None and test_rfs is not None:
        plt.figure(figsize=(4, 3))
        plt.bar(
            ["Test Acc", "Rule Fidelity"],
            [test_acc, test_rfs],
            color=["green", "orange"],
        )
        plt.ylim(0, 1)
        plt.title(f"{ds_key}: Test Accuracy vs Rule Fidelity")
        plt.savefig(
            os.path.join(working_dir, f"{ds_key}_acc_vs_rfs.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
except Exception as e:
    print(f"Error creating acc vs rfs bar: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print summary metrics
# ------------------------------------------------------------------
if test_acc is not None:
    print(f"TEST ACCURACY: {test_acc:.4f}")
if test_rfs is not None:
    print(f"TEST RULE FIDELITY: {test_rfs:.4f}")
