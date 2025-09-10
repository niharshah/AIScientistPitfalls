import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data is not None:
    epochs = range(1, len(data["losses"]["train"]) + 1)

    # 1) Loss curves ---------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, data["losses"]["train"], label="Train")
        plt.plot(epochs, data["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH – Loss Curve (Train vs Val)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) Accuracy curves ------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, data["metrics"]["train_acc"], label="Train")
        plt.plot(epochs, data["metrics"]["val_acc"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH – Accuracy Curve (Train vs Val)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve: {e}")
        plt.close()

    # 3) Final metrics bar chart ---------------------------------------------
    try:
        final_m = data["final_metrics"]
        names = list(final_m.keys())
        values = [final_m[k] for k in names]
        plt.figure()
        plt.bar(names, values, color=["steelblue", "orange", "green"])
        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        plt.title("SPR_BENCH – Final Metrics")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating final metrics plot: {e}")
        plt.close()

    # 4) Confusion matrix heatmap --------------------------------------------
    try:
        from sklearn.metrics import confusion_matrix

        y_true = data["ground_truth"]
        y_pred = data["predictions"]
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues", aspect="auto")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("SPR_BENCH – Confusion Matrix (Normalized)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # -----------------------------------------------------------------------
    # Print final metrics to console
    print("Final metrics:")
    for k, v in data["final_metrics"].items():
        print(f"{k:>10s}: {v:.4f}")
