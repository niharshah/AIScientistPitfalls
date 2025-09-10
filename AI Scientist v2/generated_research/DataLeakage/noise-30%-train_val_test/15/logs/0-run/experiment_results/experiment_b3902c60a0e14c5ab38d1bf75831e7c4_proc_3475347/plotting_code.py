import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    exp_name, ds_name = "Remove_Label_Smoothing_Loss", "SPR_BENCH"
    rec = experiment_data[exp_name][ds_name]
    epochs = rec["epochs"]
    # -------------------------------------------------------
    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, rec["losses"]["train"], label="Train")
        plt.plot(epochs, rec["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Loss Curves (Train vs Validation)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()
    # -------------------------------------------------------
    # 2) Macro-F1 curves
    try:
        plt.figure()
        plt.plot(epochs, rec["metrics"]["train"], label="Train")
        plt.plot(epochs, rec["metrics"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH – Macro-F1 Curves (Train vs Validation)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()
    # -------------------------------------------------------
    # 3) Confusion matrix
    try:
        y_true = np.array(rec["ground_truth"])
        y_pred = np.array(rec["predictions"])
        n_labels = int(max(y_true.max(), y_pred.max())) + 1
        cm = np.zeros((n_labels, n_labels), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(
            "SPR_BENCH – Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.xticks(range(n_labels))
        plt.yticks(range(n_labels))
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
    # -------------------------------------------------------
    # 4) Label distribution comparison
    try:
        gt_counts = np.bincount(y_true, minlength=n_labels)
        pred_counts = np.bincount(y_pred, minlength=n_labels)
        x = np.arange(n_labels)
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_counts, width, label="Predictions")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("SPR_BENCH – Label Distribution (GT vs Predictions)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_label_distribution.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating label distribution plot: {e}")
        plt.close()
    # -------------------------------------------------------
    # print final metrics
    print(f"Test loss : {rec.get('test_loss'):.4f}")
    print(f"Test Macro-F1 : {rec.get('test_macroF1'):.4f}")
