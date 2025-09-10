import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SingleRelGCN"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr:
    epochs = list(range(1, len(spr["losses"]["train"]) + 1))
    # ------------ Plot 1: Loss curves ------------
    try:
        plt.figure()
        plt.plot(epochs, spr["losses"]["train"], label="Train Loss")
        plt.plot(epochs, spr["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SingleRelGCN_SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------ Plot 2: Validation metrics ------------
    try:
        cwa = [m["cwa"] for m in spr["metrics"]["val"]]
        swa = [m["swa"] for m in spr["metrics"]["val"]]
        cpx = [m["cpxwa"] for m in spr["metrics"]["val"]]
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cpx, label="CpxWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH – Validation Weighted Accuracies")
        plt.legend()
        fname = os.path.join(working_dir, "SingleRelGCN_SPR_BENCH_val_metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating val-metrics plot: {e}")
        plt.close()

    # ------------ Plot 3: Test metrics ------------
    try:
        test_m = spr["metrics"]["test"]
        plt.figure()
        plt.bar(
            ["CWA", "SWA", "CpxWA"], [test_m["cwa"], test_m["swa"], test_m["cpxwa"]]
        )
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH – Test Set Metrics")
        fname = os.path.join(working_dir, "SingleRelGCN_SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating test-metrics bar chart: {e}")
        plt.close()

    # ------------ Plot 4: Confusion matrix ------------
    try:
        import itertools

        labels = sorted(set(spr["ground_truth"]))
        label2idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(spr["ground_truth"], spr["predictions"]):
            cm[label2idx[t], label2idx[p]] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
            )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH – Confusion Matrix (Test)")
        fname = os.path.join(working_dir, "SingleRelGCN_SPR_BENCH_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
