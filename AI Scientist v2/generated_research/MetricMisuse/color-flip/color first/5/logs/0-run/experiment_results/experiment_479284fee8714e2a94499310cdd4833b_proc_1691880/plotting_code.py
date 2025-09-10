import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# convenience getter
def get_ed():
    try:
        return experiment_data["BinaryCountAblation"]["SPR_BENCH"]
    except KeyError:
        return {}


ed = get_ed()

# 1) Loss curves ---------------------------------------------------------------
try:
    tr_loss = np.asarray(ed["losses"]["train"])
    vl_loss = np.asarray(ed["losses"]["val"])
    if tr_loss.size and vl_loss.size:
        epochs = np.arange(1, len(tr_loss) + 1)
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, vl_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH – Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error plotting loss curves: {e}")
    plt.close()

# 2) Validation accuracy -------------------------------------------------------
try:
    val_m = ed["metrics"]["val"]
    if val_m:
        acc = np.array([m["acc"] for m in val_m])
        epochs = np.arange(1, len(acc) + 1)
        plt.figure()
        plt.plot(epochs, acc, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH – Validation Accuracy")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error plotting validation accuracy: {e}")
    plt.close()

# 3) Weighted metrics curves ---------------------------------------------------
try:
    if val_m:
        cwa = [m["cwa"] for m in val_m]
        swa = [m["swa"] for m in val_m]
        ccwa = [m["ccwa"] for m in val_m]
        epochs = np.arange(1, len(cwa) + 1)
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, ccwa, label="CCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH – Validation Weighted Metrics")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_weighted_metrics.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error plotting weighted metrics: {e}")
    plt.close()

# 4) Test metrics bar chart ----------------------------------------------------
try:
    tst = ed.get("metrics", {}).get("test", {})
    if tst:
        names = ["ACC", "CWA", "SWA", "CCWA"]
        vals = [tst.get(k.lower(), np.nan) for k in names]
        plt.figure()
        plt.bar(names, vals, color="skyblue")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH – Test Metrics")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error plotting test metrics: {e}")
    plt.close()

# 5) Confusion matrix ----------------------------------------------------------
try:
    gt = np.asarray(ed.get("ground_truth", []))
    pr = np.asarray(ed.get("predictions", []))
    if gt.size and pr.size and gt.shape == pr.shape:
        n_cls = int(max(gt.max(), pr.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for g, p in zip(gt, pr):
            cm[g, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH – Confusion Matrix")
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error plotting confusion matrix: {e}")
    plt.close()
