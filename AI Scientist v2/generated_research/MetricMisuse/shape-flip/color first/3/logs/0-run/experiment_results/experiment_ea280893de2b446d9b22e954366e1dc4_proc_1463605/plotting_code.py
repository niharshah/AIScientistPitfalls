import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["single_layer_rgcn"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    # convenience ----------------------------------------------------------------
    train_losses = exp["losses"]["train"]
    val_losses = exp["losses"]["val"]
    tr_metrics = exp["metrics"]["train"]
    val_metrics = exp["metrics"]["val"]
    epochs = list(range(1, len(train_losses) + 1))

    # helper to safely pull a metric list ----------------------------------------
    def metric_list(split_metrics, key):
        return [m.get(key, np.nan) for m in split_metrics]

    # 1) Loss curves -------------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("spr_bench – Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) BWA curves --------------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, metric_list(tr_metrics, "BWA"), label="Train")
        plt.plot(epochs, metric_list(val_metrics, "BWA"), label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Balanced Weighted Accuracy (BWA)")
        plt.title("spr_bench – Training vs Validation BWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_bwa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating BWA curve: {e}")
        plt.close()

    # 3) CWA & SWA curves (validation) -------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, metric_list(val_metrics, "CWA"), label="CWA")
        plt.plot(epochs, metric_list(val_metrics, "SWA"), label="SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("spr_bench – Validation CWA & SWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_cwa_swa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CWA/SWA curve: {e}")
        plt.close()

    # 4) Prediction vs Ground-Truth distribution ---------------------------------
    try:
        preds = np.array(exp.get("predictions", []))
        gts = np.array(exp.get("ground_truth", []))
        if preds.size and gts.size:
            labels = sorted(set(gts) | set(preds))
            pred_counts = [np.sum(preds == l) for l in labels]
            gt_counts = [np.sum(gts == l) for l in labels]

            x = np.arange(len(labels))
            width = 0.35
            plt.figure()
            plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
            plt.bar(x + width / 2, pred_counts, width, label="Predictions")
            plt.xlabel("Label Index")
            plt.ylabel("Count")
            plt.title(
                "spr_bench – Test Set Label Distribution\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.xticks(x, labels)
            plt.legend()
            plt.savefig(os.path.join(working_dir, "spr_bench_label_distribution.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating label distribution plot: {e}")
        plt.close()
