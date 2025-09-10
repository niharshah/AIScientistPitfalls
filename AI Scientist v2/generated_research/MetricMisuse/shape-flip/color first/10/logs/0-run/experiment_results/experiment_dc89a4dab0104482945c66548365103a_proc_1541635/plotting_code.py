import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) else 0.0


def harmonic_weighted_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# ---------------------------------------------------------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    spr = experiment_data["SprBench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr is not None:
    epochs = list(range(1, len(spr["losses"]["train"]) + 1))
    train_loss = spr["losses"]["train"]
    val_loss = spr["losses"]["val"]
    hwa_vals = [d["HWA"] for d in spr["metrics"]["val"]]

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SprBench Training vs Validation Loss\nLeft: Train, Right: Validation"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SprBench_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2) HWA curve
    try:
        plt.figure()
        plt.plot(epochs, hwa_vals, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy (HWA)")
        plt.title("SprBench Validation HWA Across Epochs")
        fname = os.path.join(working_dir, "SprBench_HWA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # 3) Final test metrics bar chart
    try:
        preds = spr["predictions"]
        gts = spr["ground_truth"]
        # If sequences exist inside predictions dict inferring from metrics; rely on val seqs length
        # Recreate sequences array length same as preds if stored, else blank strings
        seqs = [""] * len(preds)
        cwa = color_weighted_accuracy(seqs, gts, preds)
        swa = shape_weighted_accuracy(seqs, gts, preds)
        hwa = harmonic_weighted_accuracy(cwa, swa)

        metrics = ["CWA", "SWA", "HWA"]
        values = [cwa, swa, hwa]
        plt.figure()
        plt.bar(metrics, values, color=["skyblue", "salmon", "limegreen"])
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.title("SprBench Final Test Metrics\nLeft: CWA, Center: SWA, Right: HWA")
        fname = os.path.join(working_dir, "SprBench_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()
