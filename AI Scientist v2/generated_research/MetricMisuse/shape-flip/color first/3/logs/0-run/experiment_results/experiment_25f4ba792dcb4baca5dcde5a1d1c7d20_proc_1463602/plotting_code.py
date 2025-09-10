import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    bench = experiment_data["sequential_only"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    bench = None

if bench:
    # ---------- common epoch axis ----------
    epochs = np.arange(1, len(bench["losses"]["train"]) + 1)

    # ---------- 1. Loss curves -------------
    try:
        plt.figure()
        plt.plot(epochs, bench["losses"]["train"], label="Train", marker="o")
        plt.plot(epochs, bench["losses"]["val"], label="Validation", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR-BENCH Loss Curves\nTraining vs Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # helper to pull metric list
    def metric_array(name, split):
        return [m[name] for m in bench["metrics"][split]]

    # ---------- 2. BWA curves -------------
    try:
        plt.figure()
        plt.plot(epochs, metric_array("BWA", "train"), label="Train", marker="o")
        plt.plot(epochs, metric_array("BWA", "val"), label="Validation", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title("SPR-BENCH BWA Curves\nTraining vs Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_BWA_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating BWA curve plot: {e}")
        plt.close()

    # ---------- 3. CWA curves -------------
    try:
        plt.figure()
        plt.plot(epochs, metric_array("CWA", "train"), label="Train", marker="o")
        plt.plot(epochs, metric_array("CWA", "val"), label="Validation", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("CWA")
        plt.title("SPR-BENCH CWA Curves\nTraining vs Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_CWA_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CWA curve plot: {e}")
        plt.close()

    # ---------- 4. StrWA curves -----------
    try:
        plt.figure()
        plt.plot(epochs, metric_array("StrWA", "train"), label="Train", marker="o")
        plt.plot(epochs, metric_array("StrWA", "val"), label="Validation", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("StrWA")
        plt.title("SPR-BENCH StrWA Curves\nTraining vs Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_StrWA_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating StrWA curve plot: {e}")
        plt.close()

    # ---------- 5. Confusion matrix -------
    try:
        preds = np.array(bench["predictions"])
        labels = np.array(bench["ground_truth"])
        n_cls = max(preds.max(), labels.max()) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(labels, preds):
            cm[t, p] += 1

        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(
            "SPR-BENCH Confusion Matrix\nLeft: Ground Truth, Right: Generated Predictions"
        )
        plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---------- print final test metrics ---
    print("Final test metrics:", bench.get("test_metrics", {}))
