import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "dual_channel" in experiment_data:
    data = experiment_data["dual_channel"]

    # ---------- 1. Loss curve ---------------------------------
    try:
        tr_epochs, tr_losses = zip(*data["losses"]["train"])
        val_epochs, val_losses = zip(*data["losses"]["val"])

        plt.figure()
        plt.plot(tr_epochs, tr_losses, label="Train")
        plt.plot(val_epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curve\nLeft: Train, Right: Val")
        plt.legend()
        fname = "loss_curve_SPR_BENCH_dual_channel.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- 2. Metric curves ------------------------------
    try:
        epochs, metr_dicts = zip(*data["metrics"]["val"])
        cwa_vals = [m["CWA"] for m in metr_dicts]
        swa_vals = [m["SWA"] for m in metr_dicts]
        pcwa_vals = [m["PCWA"] for m in metr_dicts]

        plt.figure()
        plt.plot(epochs, cwa_vals, label="CWA")
        plt.plot(epochs, swa_vals, label="SWA")
        plt.plot(epochs, pcwa_vals, label="PCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH Validation Metrics Across Epochs")
        plt.legend()
        fname = "metric_curves_SPR_BENCH_dual_channel.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating metric curves: {e}")
        plt.close()

    # ---------- 3. Final-epoch metric summary -----------------
    try:
        last_metrics = metr_dicts[-1]  # dict with CWA,SWA,PCWA
        names = list(last_metrics.keys())
        vals = list(last_metrics.values())

        plt.figure()
        plt.bar(names, vals, color=["steelblue", "orange", "green"])
        plt.ylim(0, 1)
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.ylabel("Score")
        plt.title("SPR_BENCH Final-Epoch Validation Metrics")
        fname = "final_val_metrics_SPR_BENCH_dual_channel.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating final metric bar chart: {e}")
        plt.close()

    # ---------- 4. Correct vs incorrect predictions ----------
    try:
        preds = data["predictions"]
        gts = data["ground_truth"]
        if preds and gts and len(preds) == len(gts):
            correct = sum(p == t for p, t in zip(preds, gts))
            incorrect = len(preds) - correct
            plt.figure()
            plt.bar(
                ["Correct", "Incorrect"],
                [correct, incorrect],
                color=["seagreen", "salmon"],
            )
            plt.ylabel("Count")
            plt.title(
                "SPR_BENCH Test Prediction Accuracy\nLeft: Correct, Right: Incorrect"
            )
            fname = "prediction_accuracy_SPR_BENCH_dual_channel.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating accuracy bar chart: {e}")
        plt.close()
