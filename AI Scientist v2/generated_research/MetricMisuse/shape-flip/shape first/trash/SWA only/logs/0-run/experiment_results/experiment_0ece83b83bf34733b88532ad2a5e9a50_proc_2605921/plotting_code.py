import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------- load experiment data -------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("Remove-Symbolic-Branch", {}).get("SPR_BENCH", {})

loss_tr = ed.get("losses", {}).get("train", [])
loss_val = ed.get("losses", {}).get("val", [])

acc_tr = [m.get("acc") for m in ed.get("metrics", {}).get("train", [])]
acc_val = [m.get("acc") for m in ed.get("metrics", {}).get("val", [])]

swa_tr = [m.get("swa") for m in ed.get("metrics", {}).get("train", [])]
swa_val = [m.get("swa") for m in ed.get("metrics", {}).get("val", [])]

test_metrics = ed.get("metrics", {}).get("test", {})
test_acc = test_metrics.get("acc")
test_swa = test_metrics.get("swa")

preds = np.array(ed.get("predictions", []))
gts = np.array(ed.get("ground_truth", []))

# --------------------------- plots ----------------------------------
# 1. Loss curves
try:
    plt.figure()
    plt.plot(loss_tr, label="Train")
    plt.plot(loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH – Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2. Accuracy curves
try:
    plt.figure()
    plt.plot(acc_tr, label="Train")
    plt.plot(acc_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH – Accuracy Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# 3. SWA curves
try:
    plt.figure()
    plt.plot(swa_tr, label="Train")
    plt.plot(swa_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR_BENCH – SWA Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# 4. Final test metrics bar plot
try:
    plt.figure()
    plt.bar(["Accuracy", "SWA"], [test_acc, test_swa])
    plt.ylim(0, 1)
    plt.title("SPR_BENCH – Final Test Metrics")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar plot: {e}")
    plt.close()

# 5. Confusion matrix heatmap
try:
    if preds.size and gts.size:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(gts, preds, labels=[0, 1])
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.title("SPR_BENCH – Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks([0, 1], ["0", "1"])
        plt.yticks([0, 1], ["0", "1"])
        plt.colorbar()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# --------------------- print metrics --------------------------------
print(f"TEST Accuracy: {test_acc:.4f} | TEST SWA: {test_swa:.4f}")
