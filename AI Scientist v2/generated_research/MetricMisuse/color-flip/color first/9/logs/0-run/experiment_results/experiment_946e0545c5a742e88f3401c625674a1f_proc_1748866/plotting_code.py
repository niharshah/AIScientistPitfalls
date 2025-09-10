import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data["no_glyph_clustering"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = None

if run:  # proceed only if data loaded
    # ---------- common helpers ----------
    def get_losses(split):
        arr = run["losses"][split]  # list of (lr, ep, loss)
        if not arr:
            return [], []
        ep, loss = zip(*[(t[1], t[2]) for t in arr])
        return np.array(ep), np.array(loss)

    def get_metrics():
        arr = run["metrics"]["val"]  # list of (lr, ep, cwa, swa, hwa, cna)
        if not arr:
            return {}, []
        ep = [t[1] for t in arr]
        names = ["CWA", "SWA", "HWA", "CNA"]
        vals = {n: [t[i + 2] for t in arr] for i, n in enumerate(names)}
        return vals, ep

    train_ep, train_loss = get_losses("train")
    val_ep, val_loss = get_losses("val")
    metrics_dict, met_ep = get_metrics()
    preds = np.array(run.get("predictions", []))
    gts = np.array(run.get("ground_truth", []))

    # ---------- 1. loss curve ----------
    try:
        plt.figure()
        if len(train_ep):
            plt.plot(train_ep, train_loss, label="Train")
        if len(val_ep):
            plt.plot(val_ep, val_loss, label="Validation")
        plt.title("SPR_BENCH Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- 2. validation metrics ----------
    try:
        if metrics_dict:
            plt.figure()
            for name, vals in metrics_dict.items():
                plt.plot(met_ep, vals, label=name)
            plt.title("SPR_BENCH Validation Metrics over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_metrics.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()

    # ---------- 3. label distribution ----------
    try:
        if preds.size and gts.size:
            labels = np.unique(np.concatenate([gts, preds]))
            gt_counts = [(gts == l).sum() for l in labels]
            pr_counts = [(preds == l).sum() for l in labels]
            x = np.arange(len(labels))
            width = 0.35
            plt.figure()
            plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
            plt.bar(x + width / 2, pr_counts, width, label="Predictions")
            plt.title("SPR_BENCH Label Distribution")
            plt.xlabel("Label")
            plt.ylabel("Count")
            plt.xticks(x, labels)
            plt.legend()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_label_distribution.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating label distribution plot: {e}")
        plt.close()

    # ---------- 4. confusion matrix ----------
    try:
        if preds.size and gts.size:
            cm = np.zeros((2, 2), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.title(
                "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
            )
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.xticks([0, 1], ["0", "1"])
            plt.yticks([0, 1], ["0", "1"])
            plt.colorbar()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()
