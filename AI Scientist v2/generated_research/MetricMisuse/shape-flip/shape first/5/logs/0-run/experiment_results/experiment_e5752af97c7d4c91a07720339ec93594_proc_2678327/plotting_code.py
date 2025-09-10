import matplotlib.pyplot as plt
import numpy as np
import os

# ----- paths -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load experiment data -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp_rec = experiment_data.get("SPR_BENCH", None)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp_rec = None

if exp_rec is not None:
    # -------- extract arrays --------
    train_loss = np.asarray(exp_rec["losses"]["train"])
    val_loss = np.asarray(exp_rec["losses"]["val"])
    train_swa = np.asarray(exp_rec["metrics"]["train_swa"])
    val_swa = np.asarray(exp_rec["metrics"]["val_swa"])
    preds = np.asarray(exp_rec.get("predictions", []))
    gts = np.asarray(exp_rec.get("ground_truth", []))
    epochs = np.arange(1, len(train_loss) + 1)

    # -------- plot 1: loss curves --------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------- plot 2: SWA curves --------
    try:
        plt.figure()
        plt.plot(epochs, train_swa, label="Train SWA")
        plt.plot(epochs, val_swa, label="Validation SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH: Train vs Validation SWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # -------- plot 3: confusion matrix --------
    try:
        if preds.size and gts.size:
            num_classes = int(max(preds.max(), gts.max())) + 1
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for p, g in zip(preds, gts):
                cm[g, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted Label")
            plt.ylabel("Ground Truth Label")
            plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # -------- evaluation metrics --------
    test_acc = (preds == gts).mean() if preds.size else np.nan
    print(f"Test Accuracy: {test_acc:.4f}")
else:
    print("No experiment data found.")
