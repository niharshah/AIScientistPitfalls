import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load experiment data ---------------------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

bench = exp.get("no_pretrain", {}).get("SPR_BENCH", {})

# ------------------ PLOT 1: loss curves ----------------------
try:
    train_losses = bench.get("losses", {}).get("train", [])
    val_losses = bench.get("losses", {}).get("val", [])
    if train_losses and val_losses:
        epochs = np.arange(1, len(train_losses) + 1)
        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Training vs. Validation Loss")
        plt.legend()
        fn = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
    else:
        print("Loss data missing, skipping loss plot.")
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------ PLOT 2: CCWA metric ----------------------
try:
    val_ccwa = bench.get("metrics", {}).get("val_CCWA", [])
    if val_ccwa:
        epochs = np.arange(1, len(val_ccwa) + 1)
        plt.figure()
        plt.plot(epochs, val_ccwa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("CCWA")
        plt.title("SPR_BENCH – Validation CCWA Across Epochs")
        fn = os.path.join(working_dir, "SPR_BENCH_val_CCWA.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
    else:
        print("CCWA data missing, skipping CCWA plot.")
    plt.close()
except Exception as e:
    print(f"Error creating CCWA plot: {e}")
    plt.close()

# ------------------ PLOT 3: confusion matrix -----------------
try:
    preds_all = bench.get("predictions", [])
    gts_all = bench.get("ground_truth", [])
    if preds_all and gts_all:
        preds = preds_all[-1]
        gts = gts_all[-1]
        num_classes = len(set(gts) | set(preds))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("SPR_BENCH – Confusion Matrix (Final Epoch)")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
                )
        fn = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
    else:
        print("Prediction data missing, skipping confusion matrix.")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
