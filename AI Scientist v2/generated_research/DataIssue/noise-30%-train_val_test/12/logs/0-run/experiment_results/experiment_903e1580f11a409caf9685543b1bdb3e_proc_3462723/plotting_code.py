import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    log = experiment_data["num_epochs"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

epochs = log["epochs"]
tr_loss, val_loss = log["losses"]["train"], log["losses"]["val"]
tr_f1, val_f1 = log["metrics"]["train_f1"], log["metrics"]["val_f1"]
preds, gts = np.array(log["predictions"]), np.array(log["ground_truth"])
best_ep, best_f1 = log["best_epoch"], log["best_val_f1"]

# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, tr_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Macro-F1 curves
try:
    plt.figure()
    plt.plot(epochs, tr_f1, label="Train Macro-F1")
    plt.plot(epochs, val_f1, label="Val Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH Macro-F1 Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# 3) Confusion matrix at best epoch
try:
    cm_size = int(max(preds.max(), gts.max()) + 1)  # number of classes
    cm = np.zeros((cm_size, cm_size), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"SPR_BENCH Confusion Matrix @ Best Epoch {best_ep}")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix_best_epoch.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

print(f"Best validation Macro-F1: {best_f1:.4f} at epoch {best_ep}")
