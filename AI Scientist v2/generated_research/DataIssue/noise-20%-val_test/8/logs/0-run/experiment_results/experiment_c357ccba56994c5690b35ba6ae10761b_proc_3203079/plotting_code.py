import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ----------- setup & data loading -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    exp = experiment_data["max_depth_tuning"]["SPR_BENCH"]
    depths_raw = exp["depths"]
    depths = [str(d) for d in depths_raw]
    acc = exp["metrics"]
    loss = exp["losses"]
    sefa = exp["sefa"]
    gt = np.array(exp["ground_truth"])
    best_depth = str(exp["best_depth"])
    best_preds = np.array(exp["predictions"][best_depth])

    # ------------ Plot 1: Accuracies -------------
    try:
        plt.figure()
        plt.plot(depths, acc["train"], marker="o", label="Train")
        plt.plot(depths, acc["val"], marker="o", label="Validation")
        plt.plot(depths, acc["test"], marker="o", label="Test")
        plt.title("SPR_BENCH Accuracy vs. Tree Depth")
        plt.xlabel("Max Depth")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_vs_depth.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ------------ Plot 2: Losses -----------------
    try:
        plt.figure()
        plt.plot(depths, loss["train"], marker="o", label="Train")
        plt.plot(depths, loss["val"], marker="o", label="Validation")
        plt.plot(depths, loss["test"], marker="o", label="Test")
        plt.title("SPR_BENCH Loss (1-Accuracy) vs. Tree Depth")
        plt.xlabel("Max Depth")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_vs_depth.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------ Plot 3: SEFA -------------------
    try:
        plt.figure()
        plt.plot(depths, sefa, marker="o", color="purple")
        plt.title("SPR_BENCH SEFA Score vs. Tree Depth")
        plt.xlabel("Max Depth")
        plt.ylabel("SEFA")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_sefa_vs_depth.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SEFA plot: {e}")
        plt.close()

    # ------------ Plot 4: Confusion Matrix -------
    try:
        cm = confusion_matrix(gt, best_preds)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.title(f"SPR_BENCH Confusion Matrix (Best Depth={best_depth})")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.savefig(
            os.path.join(
                working_dir, f"SPR_BENCH_confusion_matrix_depth_{best_depth}.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    print("Plotting complete. Figures saved to", working_dir)
