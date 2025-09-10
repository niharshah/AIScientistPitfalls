import matplotlib.pyplot as plt
import numpy as np
import os

# paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["min_samples_leaf_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

params = np.array(spr["param_values"])
train_acc = np.array(spr["metrics"]["train"])
val_acc = np.array(spr["metrics"]["val"])
train_loss = np.array(spr["losses"]["train"])
val_loss = np.array(spr["losses"]["val"])
y_true = np.array(spr["ground_truth"])
y_pred = np.array(spr["predictions"])

# Plot 1: accuracy curves
try:
    plt.figure()
    plt.plot(params, train_acc, marker="o", label="Train")
    plt.plot(params, val_acc, marker="s", label="Validation")
    plt.xlabel("min_samples_leaf")
    plt.ylabel("Accuracy")
    plt.title(
        "SPR_BENCH Accuracy vs min_samples_leaf\nLeft/Right: Train vs Validation Curves"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot 2: loss curves
try:
    plt.figure()
    plt.plot(params, train_loss, marker="o", label="Train")
    plt.plot(params, val_loss, marker="s", label="Validation")
    plt.xlabel("min_samples_leaf")
    plt.ylabel("Loss (1-Accuracy)")
    plt.title(
        "SPR_BENCH Loss vs min_samples_leaf\nLeft/Right: Train vs Validation Curves"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot 3: confusion matrix bar chart
try:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    labels = ["TN", "FP", "FN", "TP"]
    counts = cm.flatten()
    plt.figure()
    plt.bar(labels, counts, color="skyblue")
    for i, v in enumerate(counts):
        plt.text(i, v + 0.5, str(v), ha="center")
    plt.title(
        "SPR_BENCH Confusion Matrix on Test Set\nLeft: Ground Truth, Right: Predictions"
    )
    plt.ylabel("Count")
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# Print summary metrics
print(f"Best min_samples_leaf parameter: {spr['best_param']}")
print(f"Best validation accuracy     : {spr['best_val_acc']:.4f}")
print(f"Test accuracy                : {spr['test_acc']:.4f}")
print(f"Self-Explain Fidelity (SEFA) : {spr['SEFA']:.4f}")
