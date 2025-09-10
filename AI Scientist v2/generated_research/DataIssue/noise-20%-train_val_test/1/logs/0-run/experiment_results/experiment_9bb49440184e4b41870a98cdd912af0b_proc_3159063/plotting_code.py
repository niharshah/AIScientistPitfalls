import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -----------------------------------------------------------------------------#
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    exp = experiment_data["weight_decay"]["SPR_BENCH"]
    wds = exp["settings"]
    train_losses = exp["losses"]["train"]  # list[len(wds)][epochs]
    val_losses = exp["losses"]["val"]
    train_accs = exp["metrics"]["train"]
    val_accs = exp["metrics"]["val"]
    test_accs = exp["test_acc"]
    gts = np.array(exp["ground_truth"])

    # -------------------------------------------------------------------------#
    # 1) Loss curves
    try:
        plt.figure()
        for i, wd in enumerate(wds):
            epochs = np.arange(1, len(train_losses[i]) + 1)
            plt.plot(epochs, train_losses[i], label=f"train (wd={wd})", linestyle="-")
            plt.plot(epochs, val_losses[i], label=f"val   (wd={wd})", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training/Validation Loss vs Epoch")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_weight_decay.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------------------------------------------------------------------------#
    # 2) Accuracy curves
    try:
        plt.figure()
        for i, wd in enumerate(wds):
            epochs = np.arange(1, len(train_accs[i]) + 1)
            plt.plot(epochs, train_accs[i], label=f"train (wd={wd})", linestyle="-")
            plt.plot(epochs, val_accs[i], label=f"val   (wd={wd})", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Training/Validation Accuracy vs Epoch")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves_weight_decay.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # -------------------------------------------------------------------------#
    # 3) Final test accuracy bar chart
    try:
        plt.figure()
        plt.bar([str(wd) for wd in wds], test_accs, color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("Test Accuracy")
        plt.xlabel("Weight Decay")
        plt.title("SPR_BENCH: Test Accuracy vs Weight Decay")
        fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_weight_decay.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test-accuracy plot: {e}")
        plt.close()

    # -------------------------------------------------------------------------#
    # 4) Confusion matrix for best model
    try:
        best_idx = int(np.argmax(test_accs))
        preds = np.array(exp["predictions"][best_idx])
        num_labels = preds.max() + 1
        cm = np.zeros((num_labels, num_labels), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"SPR_BENCH Confusion Matrix (Best wd={wds[best_idx]})")
        plt.xticks(range(num_labels))
        plt.yticks(range(num_labels))
        fname = os.path.join(
            working_dir, "SPR_BENCH_confusion_matrix_best_weight_decay.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion-matrix plot: {e}")
        plt.close()

    # -------------------------------------------------------------------------#
    # Print numeric summary
    print("Weight Decay  |  Test Accuracy")
    for wd, acc in zip(wds, test_accs):
        print(f"{wd:<12}  {acc:.4f}")
