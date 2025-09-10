import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------- paths / load
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

tag, ds_key = "no_pos_emb", "SPR"
data = experiment_data.get(tag, {}).get(ds_key, {})

# --------------------------------------------------------------------- 1) loss curves
try:
    tr_losses = data.get("losses", {}).get("train", [])
    val_losses = data.get("losses", {}).get("val", [])
    epochs = list(range(1, len(tr_losses) + 1))
    if tr_losses and val_losses:
        plt.figure()
        plt.plot(epochs, tr_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Loss Curves (no_pos_emb)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_no_pos_emb_loss_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# --------------------------------------------------------------------- 2) validation accuracy metrics
try:
    val_metrics = data.get("metrics", {}).get("val", [])
    if val_metrics:
        cwa = [m["cwa"] for m in val_metrics]
        swa = [m["swa"] for m in val_metrics]
        cva = [m["cva"] for m in val_metrics]
        epochs = list(range(1, len(cwa) + 1))
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cva, label="CVA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR Validation Accuracies (no_pos_emb)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_no_pos_emb_val_accuracies.png")
        plt.savefig(fname)
        print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# --------------------------------------------------------------------- 3) confusion matrix on test set
try:
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))
    if preds.size and gts.size:
        num_cls = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for p, g in zip(preds, gts):
            cm[g, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR Confusion Matrix (no_pos_emb)")
        for i in range(num_cls):
            for j in range(num_cls):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "SPR_no_pos_emb_confusion_matrix.png")
        plt.savefig(fname)
        print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
