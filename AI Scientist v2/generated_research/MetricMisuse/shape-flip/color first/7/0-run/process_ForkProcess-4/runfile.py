import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Short-circuit if no data
if "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]
    losses_tr = data["losses"]["train"]
    losses_val = data["losses"]["val"]
    acc_tr = [d["acc"] for d in data["metrics"]["train"]]
    acc_val = [d["acc"] for d in data["metrics"]["val"]]
    cowa_tr = [d["cowa"] for d in data["metrics"]["train"]]
    cowa_val = [d["cowa"] for d in data["metrics"]["val"]]
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])
    seqs = data["sequences"]

    # helper copied from training script
    def count_color_variety(sequence: str) -> int:
        return len(
            set(token[1] for token in sequence.strip().split() if len(token) > 1)
        )

    def count_shape_variety(sequence: str) -> int:
        return len(set(token[0] for token in sequence.strip().split() if token))

    def complexity_weight(sequence: str) -> int:
        return count_color_variety(sequence) + count_shape_variety(sequence)

    def complexity_weighted_accuracy(seqs, y_true, y_pred):
        weights = [complexity_weight(s) for s in seqs]
        correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
        return float(sum(correct)) / max(1, sum(weights))

    # print final metrics
    final_acc = (preds == gts).mean()
    final_cowa = complexity_weighted_accuracy(seqs, gts, preds)
    print(f"Final Test Accuracy: {final_acc:.3f}")
    print(f"Final Test CoWA: {final_cowa:.3f}")

    # ---------------- plots ----------------
    # 1. Loss curve
    try:
        plt.figure()
        epochs = range(1, len(losses_tr) + 1)
        plt.plot(epochs, losses_tr, label="Train Loss")
        plt.plot(epochs, losses_val, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2. Accuracy & CoWA curve (two y-axes)
    try:
        plt.figure()
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(epochs, acc_tr, "g-", label="Train Acc")
        ax1.plot(epochs, acc_val, "g--", label="Val Acc")
        ax2.plot(epochs, cowa_tr, "b-", label="Train CoWA")
        ax2.plot(epochs, cowa_val, "b--", label="Val CoWA")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy", color="g")
        ax2.set_ylabel("CoWA", color="b")
        plt.title("SPR_BENCH Accuracy (green) & CoWA (blue)")
        # build combined legend
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels, loc="center right")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_acc_cowa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating acc/CoWA curve: {e}")
        plt.close()

    # 3. Confusion matrix
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(gts, preds, labels=[0, 1])
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(
            "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
else:
    print("No SPR_BENCH data found in experiment_data.npy")
