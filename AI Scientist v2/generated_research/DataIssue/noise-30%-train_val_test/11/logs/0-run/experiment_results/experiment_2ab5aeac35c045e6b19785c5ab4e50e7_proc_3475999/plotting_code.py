import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------- paths / load
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    raise RuntimeError(f"Error loading experiment data: {e}")

rec = experiment_data["NoPadMask_Transformer"]["SPR_BENCH"]
epochs = rec.get("epochs", [])
train_loss = rec["losses"].get("train", [])
val_loss = rec["losses"].get("val", [])
train_f1 = rec["metrics"].get("train_macro_f1", [])
val_f1 = rec["metrics"].get("val_macro_f1", [])
preds = rec.get("predictions", [])
trues = rec.get("ground_truth", [])
test_f1 = rec.get("test_macro_f1", None)

# --------------------------------------------------------------------- plot 1: loss curves
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Loss Curves\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# --------------------------------------------------------------------- plot 2: macro-F1 curves
try:
    plt.figure()
    plt.plot(epochs, train_f1, label="Train")
    plt.plot(epochs, val_f1, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH – Macro-F1 Curves\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_macro_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating macro-F1 curves: {e}")
    plt.close()

# --------------------------------------------------------------------- plot 3: confusion matrix (optional)
try:
    if preds and trues:
        preds_arr = np.array(preds)
        trues_arr = np.array(trues)
        num_classes = int(max(preds_arr.max(), trues_arr.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, t in zip(preds_arr, trues_arr):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"SPR_BENCH – Test Confusion Matrix\nMacro-F1={test_f1:.3f}")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "SPR_BENCH_test_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# --------------------------------------------------------------------- print metric
print(f"Test macro-F1: {test_f1}")
