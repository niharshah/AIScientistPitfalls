import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------- setup & data loading -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is None or "SPR_BENCH" not in experiment_data:
    print("No plotting performed because experiment data is missing or malformed.")
else:
    data = experiment_data["SPR_BENCH"]

    # --------------- helper for safe extraction --------------
    def unzip_tuples(tuples_list):
        if not tuples_list:
            return [], []
        epochs, vals = zip(*tuples_list)
        return list(epochs), list(vals)

    # --------------- 1) loss curve ---------------------------
    try:
        tr_epochs, tr_losses = unzip_tuples(data["losses"]["train"])
        val_epochs, val_losses = unzip_tuples(data["losses"]["val"])

        plt.figure()
        plt.plot(tr_epochs, tr_losses, label="Train")
        plt.plot(val_epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curve\nLeft: Train, Right: Validation")
        plt.legend()
        fname = "loss_curve_SPR_BENCH.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # --------------- 2) metric curves ------------------------
    try:
        val_metric_tuples = data["metrics"]["val"]
        epochs = [e for e, _ in val_metric_tuples]
        cwa = [d["CWA"] for _, d in val_metric_tuples]
        swa = [d["SWA"] for _, d in val_metric_tuples]
        pcwa = [d["PCWA"] for _, d in val_metric_tuples]

        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, pcwa, label="PCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(
            "SPR_BENCH Validation Metrics Over Epochs\nLeft: CWA, Middle: SWA, Right: PCWA"
        )
        plt.legend()
        fname = "metric_curves_SPR_BENCH.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating metric curves: {e}")
        plt.close()

    # --------------- 3) final-epoch metric bar ---------------
    try:
        final_metrics = val_metric_tuples[-1][1] if val_metric_tuples else {}
        names, values = zip(*final_metrics.items()) if final_metrics else ([], [])

        plt.figure()
        plt.bar(names, values, color=["steelblue", "orange", "green"])
        plt.ylabel("Score")
        plt.title(
            "SPR_BENCH Final-Epoch Validation Metrics\nLeft: CWA, Middle: SWA, Right: PCWA"
        )
        fname = "final_val_metrics_SPR_BENCH.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating final metric bar chart: {e}")
        plt.close()

    # --------------- 4) test accuracy bar --------------------
    try:
        preds = data.get("predictions", [])
        gts = data.get("ground_truth", [])
        correct = sum(p == g for p, g in zip(preds, gts))
        total = len(preds)
        acc = correct / total if total else 0.0

        plt.figure()
        plt.bar(
            ["Correct", "Incorrect"], [correct, total - correct], color=["green", "red"]
        )
        plt.ylabel("Count")
        plt.title(f"SPR_BENCH Test Prediction Accuracy\nAccuracy = {acc:.3f}")
        fname = "test_accuracy_SPR_BENCH.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------------- print key metrics ----------------------
    try:
        print("Final validation metrics:", final_metrics)
        print(f"Test accuracy: {acc:.4f}  ({correct}/{total})")
    except Exception as e:
        print(f"Error printing metrics: {e}")
