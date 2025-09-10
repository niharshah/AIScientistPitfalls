import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    ds_name = "SPR_BENCH"
    rec = experiment_data[ds_name]

    train_loss = rec["losses"]["train"]
    val_loss = rec["losses"]["val"]
    val_swa = rec["metrics"]["val"]
    val_aca = rec["ACA"]["val"]
    test_swa = rec["metrics"]["test_SWA"]
    test_cwa = rec["metrics"]["test_CWA"]
    test_aca = rec["ACA"]["test"]
    preds = np.array(rec["predictions"])
    gts = np.array(rec["ground_truth"])
    epochs = np.arange(1, len(train_loss) + 1)

    # ------------------------------------------------- Plot 1: loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, "r--", label="Train")
        plt.plot(epochs, val_loss, "b-", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title(f"{ds_name}: Training & Validation Loss")
        plt.legend()
        fn = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
        plt.savefig(fn)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ------------------------------------------------- Plot 2: validation SWA curves
    try:
        plt.figure()
        plt.plot(epochs, val_swa, "g-")
        plt.xlabel("Epoch")
        plt.ylabel("Validation SWA")
        plt.title(f"{ds_name}: Validation Shape-Weighted Accuracy")
        fn = os.path.join(working_dir, f"{ds_name}_val_swa_curve.png")
        plt.savefig(fn)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve plot: {e}")
        plt.close()

    # ------------------------------------------------- Plot 3: validation ACA curves
    try:
        plt.figure()
        plt.plot(epochs, val_aca, "m-")
        plt.xlabel("Epoch")
        plt.ylabel("Validation ACA")
        plt.title(f"{ds_name}: Validation Augmentation-Consistency Accuracy")
        fn = os.path.join(working_dir, f"{ds_name}_val_aca_curve.png")
        plt.savefig(fn)
        plt.close()
    except Exception as e:
        print(f"Error creating ACA curve plot: {e}")
        plt.close()

    # ------------------------------------------------- Plot 4: test metric summary
    try:
        plt.figure()
        metrics = [test_swa, test_cwa, test_aca]
        names = ["SWA", "CWA", "ACA"]
        plt.bar(names, metrics, color=["g", "c", "m"])
        plt.ylim(0, 1.0)
        for i, v in enumerate(metrics):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.ylabel("Score")
        plt.title(f"{ds_name}: Test Metrics Summary")
        fn = os.path.join(working_dir, f"{ds_name}_test_metric_summary.png")
        plt.savefig(fn)
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar plot: {e}")
        plt.close()

    # ------------------------------------------------- Plot 5: confusion matrix
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(gts, preds)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title(f"{ds_name}: Confusion Matrix\nLeft: True, Right: Predicted")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center", color="red")
        fn = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
        plt.savefig(fn)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ------------------------------------------------------------------ print final metrics
    print(
        f"Test results for {ds_name} -> SWA: {test_swa:.4f}, CWA: {test_cwa:.4f}, ACA: {test_aca:.4f}"
    )
else:
    print("No experiment data found for SPR_BENCH")
