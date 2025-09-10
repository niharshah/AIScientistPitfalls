import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------- load experiment data ----------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed is not None:
    epochs = range(1, len(ed["losses"]["train"]) + 1)

    # ---------------------------- plot 1 ----------------------------- #
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
        plt.plot(epochs, ed["losses"]["val"], label="Val Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------------------------- plot 2 ----------------------------- #
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["train_acc"], label="Train Acc")
        plt.plot(epochs, ed["metrics"]["val_acc"], label="Val Acc")
        plt.title("SPR_BENCH: Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve: {e}")
        plt.close()

    # ---------------------------- plot 3 ----------------------------- #
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["Rule_Fidelity"], color="purple")
        plt.title("SPR_BENCH: Rule Fidelity Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_rule_fidelity_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating rule fidelity curve: {e}")
        plt.close()

    # ---------------------------- plot 4 ----------------------------- #
    try:
        preds = ed["predictions"].astype(int)
        gts = ed["ground_truth"].astype(int)
        n_cls = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1

        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title(
            "SPR_BENCH: Confusion Matrix\nLeft: Ground Truth (rows)  Right: Predictions (cols)"
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------------------- print evaluation metric ------------------ #
    test_acc = (ed["predictions"] == ed["ground_truth"]).mean()
    print(f"Final Test Accuracy: {test_acc:.3f}")
