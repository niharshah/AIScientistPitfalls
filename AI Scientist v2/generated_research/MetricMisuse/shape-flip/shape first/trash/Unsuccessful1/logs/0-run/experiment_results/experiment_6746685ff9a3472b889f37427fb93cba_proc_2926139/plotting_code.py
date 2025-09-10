import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# Load experiment data                                               #
# ------------------------------------------------------------------ #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ed = experiment_data["Remove-Symbolic-Feature-Auxiliary"]["SPR_BENCH"]
    epochs = ed["epochs"]
    tr_loss = ed["losses"]["train"]
    val_loss = ed["losses"]["val"]
    swa_vals = ed["metrics"]["SWA"]["val"]
    cwa_vals = ed["metrics"]["CWA"]["val"]
    hrg_vals = ed["metrics"]["HRG"]["val"]
    test_swa = ed["metrics"]["SWA"]["test"]
    test_cwa = ed["metrics"]["CWA"]["test"]
    test_hrg = ed["metrics"]["HRG"]["test"]
    preds = np.array(ed["predictions"])
    gts = np.array(ed["ground_truth"])

    # ------------------------- FIGURE 1 ----------------------------- #
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------------------------- FIGURE 2 ----------------------------- #
    try:
        plt.figure()
        plt.plot(epochs, swa_vals, label="SWA")
        plt.plot(epochs, cwa_vals, label="CWA")
        plt.plot(epochs, hrg_vals, label="HRG")
        plt.title("SPR_BENCH Validation Metrics Over Epochs\nSWA / CWA / HRG")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.legend()
        path = os.path.join(working_dir, "SPR_BENCH_metric_curves.png")
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error creating metric curves: {e}")
        plt.close()

    # ------------------------- FIGURE 3 ----------------------------- #
    try:
        if preds.size and gts.size:
            labels = sorted(set(gts).union(set(preds)))
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xticks(labels)
            plt.yticks(labels)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.title("SPR_BENCH Confusion Matrix\nTest Set (Valid/Invalid)")
            path = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(path)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------------------- FIGURE 4 ----------------------------- #
    try:
        plt.figure()
        metrics = ["SWA", "CWA", "HRG"]
        values = [test_swa, test_cwa, test_hrg]
        plt.bar(metrics, values, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        plt.title("SPR_BENCH Final Test Metrics\nBar Plot of SWA / CWA / HRG")
        path = os.path.join(working_dir, "SPR_BENCH_test_metrics_bar.png")
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar chart: {e}")
        plt.close()

    # ------------------------- PRINT EVAL --------------------------- #
    print(
        f"Final Test Metrics -> SWA: {test_swa:.4f}, CWA: {test_cwa:.4f}, HRG: {test_hrg:.4f}"
    )
