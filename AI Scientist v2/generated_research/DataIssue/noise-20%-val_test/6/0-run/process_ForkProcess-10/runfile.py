import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load -------------------------
try:
    ed = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = ed["num_epochs"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr is not None:
    mtr = spr["metrics"]
    loss = spr["losses"]
    search_vals = spr["search_vals"]
    search_val_accs = spr["search_val_accs"]
    preds = spr["predictions"]
    gts = spr["ground_truth"]
    epochs = np.arange(1, len(mtr["train_acc"]) + 1)

    # 1. accuracy
    try:
        plt.figure()
        plt.plot(epochs, mtr["train_acc"], label="Train")
        plt.plot(epochs, mtr["val_acc"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Train vs Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 2. loss
    try:
        plt.figure()
        plt.plot(epochs, loss["train"], label="Train")
        plt.plot(epochs, loss["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 3. rule fidelity
    try:
        plt.figure()
        plt.plot(epochs, mtr["rule_fidelity"], color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title("SPR_BENCH: Rule Fidelity Across Epochs")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating rule fidelity plot: {e}")
        plt.close()

    # 4. grid-search curve
    try:
        plt.figure()
        plt.plot(search_vals, search_val_accs, marker="o")
        plt.xlabel("Max Epochs Tried")
        plt.ylabel("Dev Accuracy")
        plt.title("SPR_BENCH: Grid-Search Validation Accuracy")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_grid_search_val_acc.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating search plot: {e}")
        plt.close()

    # 5. confusion matrix
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # report test accuracy
    try:
        test_acc = (preds == gts).mean()
        print(f"Test accuracy: {test_acc:.3f}")
    except Exception as e:
        print(f"Error computing test accuracy: {e}")
