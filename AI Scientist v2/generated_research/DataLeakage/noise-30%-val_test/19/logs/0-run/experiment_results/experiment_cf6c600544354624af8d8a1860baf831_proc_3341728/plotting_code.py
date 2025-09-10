import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

models = list(experiment_data.keys())

# 1-2) Train/val MCC curves
for model in models:
    try:
        mcc_train = experiment_data[model]["metrics"]["train_MCC"]
        mcc_val = experiment_data[model]["metrics"]["val_MCC"]
        plt.figure()
        plt.plot(mcc_train, label="Train MCC")
        plt.plot(mcc_val, label="Val MCC")
        plt.xlabel("Epoch update")
        plt.ylabel("MCC")
        plt.title(f"{model} – Train vs Val MCC")
        plt.legend()
        fname = os.path.join(working_dir, f"mcc_curve_{model}.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curve for {model}: {e}")
        plt.close()

# 3-4) Confusion matrices
for model in models:
    try:
        gt = np.array(experiment_data[model]["ground_truth"])
        pred = np.array(experiment_data[model]["predictions"])
        if gt.size == 0 or pred.size == 0:
            raise ValueError("Empty predictions or labels")
        cm = np.zeros((2, 2), dtype=int)
        for g, p in zip(gt, pred):
            cm[int(g), int(p)] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues", vmin=0)
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"{model} – Confusion Matrix\nLeft: GT rows, Right: Pred cols")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fname = os.path.join(working_dir, f"confusion_matrix_{model}.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {model}: {e}")
        plt.close()

# 5) Bar chart of test MCC
try:
    test_mccs = [experiment_data[m].get("test_MCC", np.nan) for m in models]
    plt.figure()
    plt.bar(models, test_mccs, color=["tab:blue", "tab:orange"])
    for i, v in enumerate(test_mccs):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.ylabel("Test MCC")
    plt.title("Model Comparison – Test MCC")
    fname = os.path.join(working_dir, "test_mcc_comparison.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test MCC bar chart: {e}")
    plt.close()

# Print numeric summary
for m in models:
    print(
        f"{m}: Test MCC = {experiment_data[m].get('test_MCC')}, "
        f"Test F1 = {experiment_data[m].get('test_F1')}"
    )
