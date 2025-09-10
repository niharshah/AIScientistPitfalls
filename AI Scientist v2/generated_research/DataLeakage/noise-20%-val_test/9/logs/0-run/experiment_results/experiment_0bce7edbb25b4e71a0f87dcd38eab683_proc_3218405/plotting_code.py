import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------- load experiment data ------------------------- #
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    exp = experiment_data["NoGateConfidence"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    epochs = np.arange(1, len(exp["losses"]["train"]) + 1)
    # ----------------------------- plot 1 -------------------------------- #
    try:
        plt.figure()
        plt.plot(epochs, exp["losses"]["train"], label="Train")
        plt.plot(epochs, exp["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()
    # ----------------------------- plot 2 -------------------------------- #
    try:
        plt.figure()
        plt.plot(epochs, exp["metrics"]["train_acc"], label="Train")
        plt.plot(epochs, exp["metrics"]["val_acc"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()
    # ----------------------------- plot 3 -------------------------------- #
    try:
        plt.figure()
        plt.plot(epochs, exp["metrics"]["Rule_Fidelity"], color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title("SPR_BENCH Rule Fidelity Over Epochs\nScalar Gate Ablation")
        fname = os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating fidelity plot: {e}")
        plt.close()
    # ----------------------------- plot 4 -------------------------------- #
    try:
        preds = np.array(exp["predictions"])
        gts = np.array(exp["ground_truth"])
        n_cls = int(max(preds.max(), gts.max())) + 1
        conf = np.zeros((n_cls, n_cls), dtype=int)
        for p, t in zip(preds, gts):
            conf[t, p] += 1
        plt.figure()
        plt.imshow(conf, cmap="Blues", interpolation="nearest")
        plt.colorbar(label="Count")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("SPR_BENCH Confusion Matrix\nTest Set Predictions")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
    # ------------------------- evaluation metric ------------------------- #
    try:
        test_acc = (preds == gts).mean()
        print(f"Test Accuracy (recomputed): {test_acc:.4f}")
    except Exception as e:
        print(f"Error computing test accuracy: {e}")
