import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load experiment data -----------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data:
    epochs = data["epochs"]
    train_loss = data["losses"]["train"]
    val_loss = data["losses"]["val"]
    train_f1 = data["metrics"]["train_macro_f1"]
    val_f1 = data["metrics"]["val_macro_f1"]
    preds = data["predictions"]
    gts = data["ground_truth"]

    # 1. Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.title("SPR_BENCH Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.legend()
        save_path = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2. Macro-F1 curves
    try:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Val Macro-F1")
        plt.title("SPR_BENCH Macro-F1 vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        save_path = os.path.join(working_dir, "spr_bench_f1_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating f1 plot: {e}")
        plt.close()

    # 3. Confusion matrix
    try:
        cm = confusion_matrix(gts, preds)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        save_path = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ------------------- print key metrics ----------------------------------
    best_val_f1 = max(val_f1) if val_f1 else None
    test_macro_f1 = np.round(np.mean(val_f1[-1:]), 4)  # fallback if not stored
    if preds and gts:
        # recompute to be sure
        from sklearn.metrics import f1_score

        test_macro_f1 = f1_score(gts, preds, average="macro")
    print(f"Best validation Macro-F1: {best_val_f1:.4f}")
    print(f"Test Macro-F1: {test_macro_f1:.4f}")
