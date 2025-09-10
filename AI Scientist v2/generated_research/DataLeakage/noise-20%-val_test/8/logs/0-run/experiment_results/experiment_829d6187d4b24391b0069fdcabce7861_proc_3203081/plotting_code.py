import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["min_samples_split"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = None

if spr_data:
    params = spr_data["param_values"]
    train_acc = spr_data["metrics"]["train"]
    val_acc = spr_data["metrics"]["val"]
    train_loss = spr_data["losses"]["train"]
    val_loss = spr_data["losses"]["val"]
    y_true = np.array(spr_data["ground_truth"])
    y_pred = np.array(spr_data["predictions"])
    test_acc = spr_data.get("test_accuracy", None)
    sefa = spr_data.get("sefa", None)

    # 1) Accuracy curves -------------------------------------------------------
    try:
        plt.figure()
        plt.plot(params, train_acc, marker="o", label="Train")
        plt.plot(params, val_acc, marker="s", label="Validation")
        plt.xlabel("min_samples_split")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Accuracy vs min_samples_split")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_vs_param.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 2) Loss curves -----------------------------------------------------------
    try:
        plt.figure()
        plt.plot(params, train_loss, marker="o", label="Train")
        plt.plot(params, val_loss, marker="s", label="Validation")
        plt.xlabel("min_samples_split")
        plt.ylabel("Loss (1-accuracy)")
        plt.title("SPR_BENCH: Loss vs min_samples_split")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fname = os.path.join(working_dir, "SPR_BENCH_loss_vs_param.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 3) Test accuracy vs SEFA --------------------------------------------------
    try:
        if test_acc is not None and sefa is not None:
            plt.figure()
            bars = plt.bar(
                ["Test Accuracy", "SEFA"], [test_acc, sefa], color=["skyblue", "salmon"]
            )
            for b in bars:
                plt.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.01,
                    f"{b.get_height():.3f}",
                    ha="center",
                    va="bottom",
                )
            plt.ylim(0, 1.05)
            plt.title("SPR_BENCH: Test Accuracy vs SEFA")
            fname = os.path.join(working_dir, "SPR_BENCH_testacc_vs_sefa.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating accuracy/SEFA bar plot: {e}")
        plt.close()

    # 4) Confusion matrix -------------------------------------------------------
    try:
        if y_pred.size and y_true.size:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_true, y_pred)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()
