import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

exp = experiment_data.get("no_latent_glyph_clustering", {}).get("SPR_BENCH", {})
loss_tr = exp.get("losses", {}).get("train", [])
loss_val = exp.get("losses", {}).get("val", [])
metrics_val = exp.get("metrics", {}).get("val", [])
preds = exp.get("predictions", [])
gts = exp.get("ground_truth", [])

epochs = range(1, len(loss_tr) + 1)

# ---------- 1) Loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train Loss")
    plt.plot(epochs, loss_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH – Train vs. Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- 2) Validation accuracy ----------
try:
    acc = [m["acc"] for m in metrics_val]
    plt.figure()
    plt.plot(epochs, acc, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH – Validation Accuracy")
    fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- 3) CWA/SWA/CompWA ----------
try:
    cwa = [m["CWA"] for m in metrics_val]
    swa = [m["SWA"] for m in metrics_val]
    comp = [m["CompWA"] for m in metrics_val]
    plt.figure()
    plt.plot(epochs, cwa, label="CWA")
    plt.plot(epochs, swa, label="SWA")
    plt.plot(epochs, comp, label="CompWA")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("SPR_BENCH – CWA / SWA / CompWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_weighted_accuracies.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating weighted accuracy plot: {e}")
    plt.close()

# ---------- 4) Confusion matrix ----------
try:
    if preds and gts:
        labels = sorted(set(gts))
        cm = np.zeros((len(labels), len(labels)), int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(labels)
        plt.yticks(labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH – Confusion Matrix")
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
