import matplotlib.pyplot as plt
import numpy as np
import os

# --------- setup ---------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load data ---------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("num_epochs_tuning", {}).get("SPR_BENCH", {})
metrics, losses = ed.get("metrics", {}), ed.get("losses", {})
train_acc, val_acc = np.array(metrics.get("train_acc", [])), np.array(
    metrics.get("val_acc", [])
)
train_loss, val_loss = np.array(losses.get("train", [])), np.array(
    losses.get("val", [])
)
rule_fid = np.array(metrics.get("rule_fidelity", []))
preds, gts = np.array(ed.get("predictions", [])), np.array(ed.get("ground_truth", []))
epochs = np.arange(1, len(train_acc) + 1)


# --------- plotting helpers ---------
def save_fig(fig, name):
    fig.savefig(os.path.join(working_dir, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


# 1. Accuracy curve
try:
    fig = plt.figure()
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy vs Epochs")
    plt.legend()
    save_fig(fig, "SPR_BENCH_train_val_accuracy.png")
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2. Loss curve
try:
    fig = plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss vs Epochs")
    plt.legend()
    save_fig(fig, "SPR_BENCH_train_val_loss.png")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3. Rule fidelity curve
try:
    fig = plt.figure()
    plt.plot(epochs, rule_fid, color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Rule Fidelity")
    plt.title("SPR_BENCH Rule Fidelity vs Epochs")
    save_fig(fig, "SPR_BENCH_rule_fidelity.png")
except Exception as e:
    print(f"Error creating rule fidelity plot: {e}")
    plt.close()

# 4. Confusion matrix
try:
    if preds.size and gts.size:
        num_classes = max(preds.max(), gts.max()) + 1
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        fig = plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        save_fig(fig, "SPR_BENCH_confusion_matrix.png")
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# --------- evaluation summary ---------
if preds.size and gts.size:
    test_accuracy = (preds == gts).mean()
    print(f"Test Accuracy: {test_accuracy:.4f}")
best_epoch = ed.get("best_epoch", None)
print(f"Best Epoch recorded: {best_epoch}")
