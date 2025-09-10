import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

boe = experiment_data.get("bag_of_embeddings", {}).get("SPR_BENCH", {})

# ---------- helper to get losses / metrics ----------
loss_tr = np.array(boe.get("losses", {}).get("train", []))  # (ep, loss)
loss_val = np.array(boe.get("losses", {}).get("val", []))
metrics_val = np.array(boe.get("metrics", {}).get("val", []))  # (ep, swa,cwa,hwa)
preds = np.array(boe.get("predictions", []))
truth = np.array(boe.get("ground_truth", []))

# ---------- Figure 1: loss curves ----------
try:
    if loss_tr.size and loss_val.size:
        plt.figure()
        plt.plot(loss_tr[:, 0], loss_tr[:, 1], label="Train")
        plt.plot(loss_val[:, 0], loss_val[:, 1], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Bag-of-Embeddings\nTraining vs Validation Loss")
        plt.legend()
        fp = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fp)
        plt.close()
        print("Saved", fp)
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- Figure 2: validation metrics ----------
try:
    if metrics_val.size:
        plt.figure()
        plt.plot(metrics_val[:, 0], metrics_val[:, 1], "o-", label="SWA")
        plt.plot(metrics_val[:, 0], metrics_val[:, 2], "s-", label="CWA")
        plt.plot(metrics_val[:, 0], metrics_val[:, 3], "^-", label="HWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Bag-of-Embeddings\nValidation Weighted Accuracies")
        plt.legend()
        fp = os.path.join(working_dir, "SPR_BENCH_weighted_accuracies.png")
        plt.savefig(fp)
        plt.close()
        print("Saved", fp)
except Exception as e:
    print(f"Error creating metrics curve: {e}")
    plt.close()

# ---------- Figure 3: confusion matrix ----------
try:
    if preds.size and truth.size:
        labels = sorted(set(truth))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(truth, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH Bag-of-Embeddings\nConfusion Matrix (Dev Set)")
        fp = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fp)
        plt.close()
        print("Saved", fp)
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- print final harmonic weighted accuracy ----------
if metrics_val.size:
    final_hwa = metrics_val[-1, 3]
    print(f"Final Harmonic Weighted Accuracy (dev): {final_hwa:.4f}")
