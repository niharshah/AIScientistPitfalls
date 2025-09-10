import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------ load experiment data -------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("count_only", {}).get("spr_bench", {})

epochs = ed.get("epochs", [])
train_loss = ed.get("losses", {}).get("train", [])
val_loss = ed.get("losses", {}).get("val", [])
train_f1 = ed.get("metrics", {}).get("train_f1", [])
val_f1 = ed.get("metrics", {}).get("val_f1", [])
test_f1 = ed.get("metrics", {}).get("test_f1", None)
preds = ed.get("predictions", [])
gts = ed.get("ground_truth", [])

# ----------------------------- plots -----------------------------------
# 1. Loss curve
try:
    if epochs and train_loss and val_loss:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2. F1 curve
try:
    if epochs and train_f1 and val_f1:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train F1")
        plt.plot(epochs, val_f1, label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("SPR_BENCH: Training vs Validation Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_f1_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# 3. Test vs Best-Val F1 bar
try:
    if test_f1 is not None and val_f1:
        best_val = max(val_f1)
        plt.figure()
        plt.bar(["Best Val", "Test"], [best_val, test_f1], color=["skyblue", "salmon"])
        for i, v in enumerate([best_val, test_f1]):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Best Validation vs Test Macro-F1")
        fname = os.path.join(working_dir, "spr_bench_f1_bar.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating F1 bar: {e}")
    plt.close()

# 4. Confusion matrix
try:
    if preds and gts:
        import itertools

        labels = sorted(list(set(gts)))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for p, g in zip(preds, gts):
            cm[g][p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(labels)
        plt.yticks(labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        # annotate
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ----------------------------- log -------------------------------------
if test_f1 is not None:
    print(f"Best Val F1: {max(val_f1):.4f} | Test F1: {test_f1:.4f}")
