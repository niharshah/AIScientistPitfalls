import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# -------- helper to fetch nested dict safely ------------
def get_rec(exp_dict, model="NoPositionalEmbedding", dataset="SPR_BENCH"):
    try:
        return exp_dict[model][dataset]
    except KeyError:
        return None


rec = get_rec(experiment_data)
if rec is None:
    print("No experiment record found; nothing to plot.")
else:
    epochs = rec.get("epochs", [])
    losses_tr = rec.get("losses", {}).get("train", [])
    losses_val = rec.get("losses", {}).get("val", [])
    swa_val = rec.get("metrics", {}).get("val", [])
    swa_test = rec.get("metrics", {}).get("test", None)
    preds = np.array(rec.get("predictions", []))
    gts = np.array(rec.get("ground_truth", []))

    # ------------- 1. Loss curves ------------------------
    try:
        if epochs and losses_tr and losses_val:
            plt.figure()
            plt.plot(epochs, losses_tr, label="Train Loss")
            plt.plot(epochs, losses_val, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("SPR_BENCH – Training vs Validation Loss")
            plt.legend()
            out_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
            plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    # ------------- 2. Validation SWA ---------------------
    try:
        if epochs and swa_val:
            plt.figure()
            plt.plot(epochs, swa_val, marker="o", label="Val SWA")
            if swa_test is not None:
                plt.scatter(
                    epochs[-1], swa_test, color="red", label=f"Test SWA={swa_test:.3f}"
                )
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title("SPR_BENCH – Validation SWA across Epochs")
            plt.legend()
            out_path = os.path.join(working_dir, "SPR_BENCH_SWA_curves.png")
            plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # ------------- 3. Confusion Matrix -------------------
    try:
        if preds.size and gts.size:
            n_classes = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((n_classes, n_classes), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("SPR_BENCH – Confusion Matrix (Test Split)")
            # annotate cells
            for i in range(n_classes):
                for j in range(n_classes):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            out_path = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()
