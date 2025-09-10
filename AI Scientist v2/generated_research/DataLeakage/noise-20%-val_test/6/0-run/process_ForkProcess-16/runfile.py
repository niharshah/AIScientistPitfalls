import matplotlib.pyplot as plt
import numpy as np
import os

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    data = experiment_data["adam_beta2"]["SPR_BENCH"]
    betas = data["beta2_values"]
    train_acc_all = data["metrics"]["train_acc"]
    val_acc_all = data["metrics"]["val_acc"]
    rule_all = data["metrics"]["rule_fidelity"]
    train_loss_all = data["losses"]["train"]
    val_loss_all = data["losses"]["val"]
    best_beta2 = data["best_beta2"]
    best_idx = betas.index(best_beta2) if best_beta2 in betas else 0

    epochs = range(1, len(train_acc_all[0]) + 1)

    # 1) Accuracy curves
    try:
        plt.figure()
        for i, beta in enumerate(betas):
            plt.plot(epochs, train_acc_all[i], label=f"train β2={beta}")
            plt.plot(epochs, val_acc_all[i], linestyle="--", label=f"val β2={beta}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Training/Validation Accuracy across β2 values")
        plt.legend(fontsize="small")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 2) Rule fidelity curves
    try:
        plt.figure()
        for i, beta in enumerate(betas):
            plt.plot(epochs, rule_all[i], label=f"β2={beta}")
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title("SPR_BENCH Rule-Fidelity across β2 values")
        plt.legend(fontsize="small")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_rule_fidelity_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating rule-fidelity plot: {e}")
        plt.close()

    # 3) Loss curves for best β2
    try:
        plt.figure()
        plt.plot(epochs, train_loss_all[best_idx], label="Train Loss")
        plt.plot(epochs, val_loss_all[best_idx], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH Loss Curves (best β2={best_beta2})")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, f"SPR_BENCH_best_beta2_{best_beta2}_loss_curves.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 4) Confusion matrix on test set
    try:
        y_true = np.array(data["ground_truth"])
        y_pred = np.array(data["predictions"])
        n_cls = int(max(y_true.max(), y_pred.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH Test Confusion Matrix")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_confusion_matrix.png"))
        plt.close()

        test_acc = (y_true == y_pred).mean()
        print(f"Test accuracy (from saved predictions) = {test_acc:.3f}")
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()
