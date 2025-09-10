import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = exp["max_n_gram_length"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment_data: {e}")
    spr = None

if spr:
    n_runs = {k: v for k, v in spr.items() if k.startswith("n=")}
    # 1) ACCURACY CURVES
    try:
        plt.figure()
        for run_name, run in n_runs.items():
            epochs = np.arange(1, len(run["metrics"]["train_acc"]) + 1)
            plt.plot(epochs, run["metrics"]["train_acc"], label=f"{run_name} train")
            plt.plot(
                epochs,
                run["metrics"]["val_acc"],
                linestyle="--",
                label=f"{run_name} val",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy Curves\nLeft: Train  Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 2) LOSS CURVES
    try:
        plt.figure()
        for run_name, run in n_runs.items():
            epochs = np.arange(1, len(run["losses"]["train"]) + 1)
            plt.plot(epochs, run["losses"]["train"], label=f"{run_name} train")
            plt.plot(
                epochs, run["losses"]["val"], linestyle="--", label=f"{run_name} val"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train  Right: Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 3) RULE FIDELITY
    try:
        plt.figure()
        for run_name, run in n_runs.items():
            epochs = np.arange(1, len(run["metrics"]["rule_fidelity"]) + 1)
            plt.plot(epochs, run["metrics"]["rule_fidelity"], label=run_name)
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title("SPR_BENCH Rule Fidelity Across Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating fidelity plot: {e}")
        plt.close()

    # 4) BEST VAL ACCURACIES BAR CHART
    try:
        plt.figure()
        names, best_vals = [], []
        for run_name, run in n_runs.items():
            names.append(run_name)
            best_vals.append(max(run["metrics"]["val_acc"]))
        plt.bar(names, best_vals, color="skyblue")
        plt.ylabel("Best Validation Accuracy")
        plt.title("SPR_BENCH Best Val Accuracy per n-gram Length")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_best_val_acc_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()

    # 5) CONFUSION MATRIX FOR BEST MODEL
    try:
        preds = spr["predictions"]
        gts = spr["ground_truth"]
        num_classes = len(np.unique(gts))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="black", fontsize=7
                )
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
