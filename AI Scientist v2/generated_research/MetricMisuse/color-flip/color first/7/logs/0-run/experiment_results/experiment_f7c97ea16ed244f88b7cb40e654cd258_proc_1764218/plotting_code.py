import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------------------------------------------------------
# 2. Identify ablation & dataset keys (only first found is used)
if experiment_data:
    abl = next(iter(experiment_data))
    dset = next(iter(experiment_data[abl]))
    data = experiment_data[abl][dset]
    epochs = data.get("epochs", [])
    train_losses = data.get("losses", {}).get("train", [])
    tr_metrics = data.get("metrics", {}).get("train", [])
    val_metrics = data.get("metrics", {}).get("val", [])
    preds = data.get("predictions", [])
    gts = data.get("ground_truth", [])
else:
    abl = dset = ""
    epochs = train_losses = tr_metrics = val_metrics = preds = gts = []


# Helper to pull metric list safely
def metric_list(metrics, key):
    return [m[key] for m in metrics] if metrics else []


# ---------------------------------------------------------------
# 3. Plotting
# 3.1 Train loss
try:
    if epochs and train_losses:
        plt.figure()
        plt.plot(epochs, train_losses, marker="o")
        plt.title(f"{dset} – {abl}\nTraining Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        fname = f"{dset}_{abl}_train_loss.png"
        plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3.2 Complexity-weighted accuracy
try:
    if epochs:
        plt.figure()
        plt.plot(epochs, metric_list(tr_metrics, "cpx"), label="Train", marker="o")
        plt.plot(epochs, metric_list(val_metrics, "cpx"), label="Val", marker="s")
        plt.title(f"{dset} – {abl}\nComplexity-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("CpxWA")
        plt.legend()
        fname = f"{dset}_{abl}_cpxwa.png"
        plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating CpxWA plot: {e}")
    plt.close()

# 3.3 Color-weighted accuracy
try:
    if epochs:
        plt.figure()
        plt.plot(epochs, metric_list(tr_metrics, "cwa"), label="Train", marker="o")
        plt.plot(epochs, metric_list(val_metrics, "cwa"), label="Val", marker="s")
        plt.title(f"{dset} – {abl}\nColor-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("CWA")
        plt.legend()
        fname = f"{dset}_{abl}_cwa.png"
        plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# 3.4 Shape-weighted accuracy
try:
    if epochs:
        plt.figure()
        plt.plot(epochs, metric_list(tr_metrics, "swa"), label="Train", marker="o")
        plt.plot(epochs, metric_list(val_metrics, "swa"), label="Val", marker="s")
        plt.title(f"{dset} – {abl}\nShape-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        fname = f"{dset}_{abl}_swa.png"
        plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# 3.5 Confusion matrix of best validation predictions
try:
    if preds and gts:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(gts, preds, labels=sorted(set(gts)))
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title(f"{dset} – {abl}\nConfusion Matrix (Val)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = f"{dset}_{abl}_confusion_matrix.png"
        plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

print("Finished plotting; files saved to", working_dir)
