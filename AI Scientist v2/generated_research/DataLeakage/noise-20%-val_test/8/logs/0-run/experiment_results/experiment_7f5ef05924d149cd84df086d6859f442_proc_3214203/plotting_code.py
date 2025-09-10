import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- LOAD DATA -------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    from sklearn.metrics import confusion_matrix

    runs = ["baseline", "length_normalized"]
    val_losses, test_accs = [], []
    y_preds, y_true = {}, {}

    for r in runs:
        run_data = experiment_data[r]["SPR_BENCH"]
        val_losses.append(run_data["losses"]["val"][0])
        test_accs.append(run_data["metrics"]["test"][0])
        y_preds[r] = np.array(run_data["predictions"])
        y_true[r] = np.array(run_data["ground_truth"])

    # -------------- PLOT 1: Validation Loss --------------
    try:
        plt.figure(figsize=(4, 4))
        plt.bar(runs, val_losses, color=["steelblue", "orange"])
        plt.ylabel("Validation Loss")
        plt.title("SPR_BENCH Validation Loss\n(Decision Tree)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating val_loss plot: {e}")
        plt.close()

    # -------------- PLOT 2: Test Accuracy --------------
    try:
        plt.figure(figsize=(4, 4))
        plt.bar(runs, test_accs, color=["seagreen", "salmon"])
        plt.ylabel("Test Accuracy")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH Test Accuracy\n(Decision Tree)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # -------------- PLOT 3: Confusion Matrix - Baseline --------------
    try:
        cm = confusion_matrix(y_true["baseline"], y_preds["baseline"])
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("SPR_BENCH Confusion Matrix - Baseline")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_cm_baseline.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating baseline CM: {e}")
        plt.close()

    # -------------- PLOT 4: Confusion Matrix - Length Normalized --------------
    try:
        cm = confusion_matrix(y_true["length_normalized"], y_preds["length_normalized"])
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("SPR_BENCH Confusion Matrix - Length Normalized")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_cm_length_normalized.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating length_normalized CM: {e}")
        plt.close()
