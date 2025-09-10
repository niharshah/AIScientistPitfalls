import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    exp = experiment_data["gcn_hidden_dim"]["SPR_BENCH"]
    hidden_dims = np.array(exp["hidden_dims"])
    train_bwa = np.array(exp["metrics"]["train"])
    val_bwa = np.array(exp["metrics"]["val"])
    train_loss = np.array(exp["losses"]["train"])
    val_loss = np.array(exp["losses"]["val"])
    preds = np.array(exp["predictions"])
    gts = np.array(exp["ground_truth"])

    # ------------------------------------------------------------------
    # Plot 1: BWA vs hidden dim
    try:
        plt.figure()
        plt.plot(hidden_dims, train_bwa, marker="o", label="Train BWA")
        plt.plot(hidden_dims, val_bwa, marker="s", label="Validation BWA")
        plt.xlabel("GCN Hidden Dimension")
        plt.ylabel("BWA")
        plt.title("SPR_BENCH: Train/Val BWA vs GCN Hidden Dim")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_BWA_vs_hidden_dim.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating BWA plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Loss vs hidden dim
    try:
        plt.figure()
        plt.plot(hidden_dims, train_loss, marker="o", label="Train Loss")
        plt.plot(hidden_dims, val_loss, marker="s", label="Validation Loss")
        plt.xlabel("GCN Hidden Dimension")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train/Val Loss vs GCN Hidden Dim")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_Loss_vs_hidden_dim.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Loss plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Plot 3: Confusion matrix on test set for best model
    try:
        num_labels = int(max(preds.max(), gts.max())) + 1
        cm = np.zeros((num_labels, num_labels), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        # normalize rows
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

        plt.figure(figsize=(5, 4))
        im = plt.imshow(cm_norm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("SPR_BENCH Test Confusion Matrix (Best Model)")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_Confusion_Matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Confusion Matrix plot: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Print overall test accuracy
    test_acc = (preds == gts).mean() if len(preds) else 0.0
    print(f"Overall test accuracy (best model): {test_acc:.4f}")
