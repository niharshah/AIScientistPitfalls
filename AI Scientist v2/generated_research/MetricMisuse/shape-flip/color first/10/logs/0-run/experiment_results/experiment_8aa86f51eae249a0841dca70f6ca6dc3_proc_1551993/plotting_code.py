import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- setup --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    run = experiment_data["SPR_BENCH"]

    # ---------- 1. loss curves ----------
    try:
        train_ep, train_loss = zip(*run["losses"]["train"])
        val_ep, val_loss = zip(*run["losses"]["val"])
        plt.figure()
        plt.plot(train_ep, train_loss, label="Train Loss")
        plt.plot(val_ep, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = "spr_bench_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- 2. validation HWA ----------
    try:
        hwa_vals = [m["hwa"] for m in run["metrics"]["val"]]
        epochs = np.arange(1, len(hwa_vals) + 1)
        plt.figure()
        plt.plot(epochs, hwa_vals, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Acc")
        plt.title("SPR_BENCH: Validation HWA")
        fname = "spr_bench_val_hwa_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # ---------- 3. test metric bar chart ----------
    try:
        tm = run["metrics"]["test"]
        names = ["CWA", "SWA", "CpxWA", "HWA"]
        vals = [tm["cwa"], tm["swa"], tm["cpx"], tm["hwa"]]
        plt.figure()
        plt.bar(names, vals, color="steelblue")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Test Metrics")
        fname = "spr_bench_test_metrics_bar.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar chart: {e}")
        plt.close()

    # ---------- 4. confusion matrix ----------
    try:
        preds = run["predictions"]
        gts = run["ground_truth"]
        labels = sorted(set(gts) | set(preds))
        lbl2idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for g, p in zip(gts, preds):
            cm[lbl2idx[g], lbl2idx[p]] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8,
                )
        plt.tight_layout()
        fname = "spr_bench_confusion_matrix.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------- print test metrics ----------
    try:
        print(
            f"TEST  CWA={tm['cwa']:.3f}  SWA={tm['swa']:.3f}  "
            f"CpxWA={tm['cpx']:.3f}  HWA={tm['hwa']:.3f}"
        )
    except Exception as e:
        print(f"Error printing test metrics: {e}")
