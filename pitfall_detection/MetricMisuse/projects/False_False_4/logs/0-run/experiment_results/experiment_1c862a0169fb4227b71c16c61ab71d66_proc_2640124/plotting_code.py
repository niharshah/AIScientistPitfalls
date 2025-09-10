import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["multi_synth_train"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    epochs = np.arange(1, len(ed["losses"]["train"]) + 1)

    # -------- helper for confusion matrix ----------
    def plot_conf_mat(y_true, y_pred, title, fname):
        classes = np.arange(max(max(y_true), max(y_pred)) + 1)
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(classes)
        plt.yticks(classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()

    # -------- 1) Loss curve ----------
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train")
        plt.plot(epochs, ed["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("spr_bench Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------- 2) Accuracy curve ----------
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["train"], label="Train")
        plt.plot(epochs, ed["metrics"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("spr_bench Accuracy Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # -------- 3) Shape-weighted accuracy curve ----------
    try:
        plt.figure()
        plt.plot(epochs, ed["swa"]["train"], label="Train")
        plt.plot(epochs, ed["swa"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("spr_bench Shape-Weighted Accuracy Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_swa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # -------- 4) Confusion matrix (validation) ----------
    try:
        if "val" in ed["predictions"] and "val" in ed["ground_truth"]:
            plot_conf_mat(
                ed["ground_truth"]["val"],
                ed["predictions"]["val"],
                "spr_bench Confusion Matrix (Validation)",
                "spr_bench_confmat_val.png",
            )
    except Exception as e:
        print(f"Error creating validation confusion matrix: {e}")
        plt.close()

    # -------- 5) Confusion matrix (test) ----------
    try:
        if "test" in ed["predictions"] and "test" in ed["ground_truth"]:
            plot_conf_mat(
                ed["ground_truth"]["test"],
                ed["predictions"]["test"],
                "spr_bench Confusion Matrix (Test)",
                "spr_bench_confmat_test.png",
            )
    except Exception as e:
        print(f"Error creating test confusion matrix: {e}")
        plt.close()

    # -------- print test metrics ----------
    if "test_metrics" in ed:
        print(f"Test metrics: {ed['test_metrics']}")
