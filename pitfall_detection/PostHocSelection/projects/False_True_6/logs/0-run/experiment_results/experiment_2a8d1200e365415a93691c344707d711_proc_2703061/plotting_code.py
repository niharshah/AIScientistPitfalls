import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment results ----------
try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp_data = None

if exp_data:
    # drill down to record
    exp_rec = exp_data["EPOCH_TUNING"]["SPR_BENCH"]
    ds_name = "SPR_BENCH"
    epochs = np.arange(1, len(exp_rec["losses"]["train"]) + 1)

    # ---------- figure 1: loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, exp_rec["losses"]["train"], label="Train")
        plt.plot(epochs, exp_rec["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- figure 2: validation accuracy ----------
    try:
        val_acc = [m["acc"] for m in exp_rec["metrics"]["val"]]
        plt.figure()
        plt.plot(epochs, val_acc, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.title(f"{ds_name} Validation Accuracy over Epochs")
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds_name}_val_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------- figure 3: test metric bar chart ----------
    try:
        test_metrics = exp_rec["metrics"]["test"]
        labels = ["acc", "swa", "cwa", "nrgs"]
        values = [test_metrics[k] for k in labels]
        plt.figure()
        plt.bar(labels, values, color="skyblue")
        plt.ylim(0, 1)
        plt.title(f"{ds_name} Test Metrics\nLeft→Right: Acc, SWA, CWA, NRGS")
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds_name}_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar chart: {e}")
        plt.close()

    # ---------- figure 4: confusion matrix ----------
    try:
        y_true = np.array(exp_rec["ground_truth"])
        y_pred = np.array(exp_rec["predictions"])
        num_classes = len(set(y_true) | set(y_pred))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{ds_name} Confusion Matrix")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------- print test metrics ----------
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
