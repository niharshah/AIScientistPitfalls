import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load data -----------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("SPR_BENCH", {})
epochs = spr.get("epochs", [])
train_loss = spr.get("losses", {}).get("train", [])
val_loss = spr.get("losses", {}).get("val", [])
train_f1 = spr.get("metrics", {}).get("train_macro_f1", [])
val_f1 = spr.get("metrics", {}).get("val_macro_f1", [])
preds = np.array(spr.get("predictions", []))
trues = np.array(spr.get("ground_truth", []))

# ---- PLOT 1: loss curves -------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nTrain vs Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---- PLOT 2: macro-F1 curves ---------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_f1, label="Train")
    plt.plot(epochs, val_f1, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH Macro-F1 Curves\nTrain vs Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_macro_f1_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# ---- PLOT 3: confusion matrix --------------------------------------------------------
try:
    if preds.size and trues.size:
        num_classes = int(max(preds.max(), trues.max())) + 1
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(trues, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---- PLOT 4: label distribution ------------------------------------------------------
try:
    if preds.size and trues.size:
        num_classes = int(max(preds.max(), trues.max())) + 1
        idx = np.arange(num_classes)
        width = 0.35
        plt.figure()
        plt.bar(
            idx - width / 2,
            np.bincount(trues, minlength=num_classes),
            width,
            label="Ground Truth",
        )
        plt.bar(
            idx + width / 2,
            np.bincount(preds, minlength=num_classes),
            width,
            label="Predictions",
        )
        plt.xlabel("Label ID")
        plt.ylabel("Count")
        plt.title("SPR_BENCH Label Distribution\nGround Truth vs Predictions")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_label_distribution.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating label distribution: {e}")
    plt.close()

# ---- evaluation metric ---------------------------------------------------------------
try:
    from sklearn.metrics import f1_score

    if preds.size and trues.size:
        print("Test Macro-F1:", f1_score(trues, preds, average="macro"))
except Exception as e:
    print(f"Could not compute F1: {e}")
