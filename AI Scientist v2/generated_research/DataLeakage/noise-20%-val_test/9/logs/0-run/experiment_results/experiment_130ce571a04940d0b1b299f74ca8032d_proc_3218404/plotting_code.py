import matplotlib.pyplot as plt
import numpy as np
import os

# ensure working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["rule_only"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    epochs = np.arange(1, len(exp["losses"]["train"]) + 1)

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, exp["losses"]["train"], label="Train Loss")
        plt.plot(epochs, exp["losses"]["val"], label="Val Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2) Accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, exp["metrics"]["train_acc"], label="Train Acc")
        plt.plot(epochs, exp["metrics"]["val_acc"], label="Val Acc")
        plt.title("SPR_BENCH: Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 3) Rule fidelity curve
    try:
        plt.figure()
        plt.plot(epochs, exp["metrics"]["Rule_Fidelity"], marker="o")
        plt.title("SPR_BENCH: Rule Fidelity Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.ylim(0, 1.05)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating fidelity plot: {e}")
        plt.close()

    # 4) Confusion matrix
    try:
        preds = exp["predictions"]
        gts = exp["ground_truth"]
        n_cls = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # print final test accuracy
    try:
        test_acc = (preds == gts).mean()
        print(f"Test Accuracy: {test_acc:.4f}")
    except Exception as e:
        print(f"Error computing test accuracy: {e}")
