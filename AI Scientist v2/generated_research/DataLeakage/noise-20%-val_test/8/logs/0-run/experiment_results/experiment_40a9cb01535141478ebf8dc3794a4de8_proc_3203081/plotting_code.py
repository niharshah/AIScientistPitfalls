import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- Path & data loading -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset_key = "SPR_BENCH"
exp_key = "splitter_tuning"
if exp_key in experiment_data and dataset_key in experiment_data[exp_key]:
    data_dict = experiment_data[exp_key][dataset_key]
else:
    print("Required keys not found in experiment_data, aborting plots.")
    data_dict = None

if data_dict:
    splitters = data_dict.get("splitter_options", [])
    x = np.arange(len(splitters))

    # ------------- Accuracy plot -----------------
    try:
        train_acc = data_dict["metrics"]["train"]
        val_acc = data_dict["metrics"]["val"]
        plt.figure()
        plt.plot(x, train_acc, "o-", label="Train Accuracy")
        plt.plot(x, val_acc, "s--", label="Validation Accuracy")
        plt.xticks(x, splitters)
        plt.ylabel("Accuracy")
        plt.title(f"SPR_BENCH Accuracy vs Splitter")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_vs_splitter.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ------------- Loss plot -----------------
    try:
        train_loss = data_dict["losses"]["train"]
        val_loss = data_dict["losses"]["val"]
        plt.figure()
        plt.plot(x, train_loss, "o-", label="Train Loss")
        plt.plot(x, val_loss, "s--", label="Validation Loss")
        plt.xticks(x, splitters)
        plt.ylabel("Loss")
        plt.title(f"SPR_BENCH Loss vs Splitter")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_vs_splitter.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------- Test metrics bar chart -------------
    try:
        plt.figure()
        bars = ["Test Accuracy", "SEFA"]
        values = [data_dict["test_accuracy"], data_dict["sefa"]]
        plt.bar(bars, values, color=["skyblue", "orange"])
        plt.ylim(0, 1.0)
        plt.title("SPR_BENCH Final Test Metrics")
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating test metric plot: {e}")
        plt.close()

    # ------------- Confusion matrix heat-map -----------
    try:
        preds = np.array(data_dict["predictions"])
        gts = np.array(data_dict["ground_truth"])
        cm = np.zeros((2, 2), dtype=int)
        for g, p in zip(gts, preds):
            cm[g, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Label (Right)")
        plt.ylabel("Ground Truth (Left)")
        plt.title(
            "SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted Labels"
        )
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
