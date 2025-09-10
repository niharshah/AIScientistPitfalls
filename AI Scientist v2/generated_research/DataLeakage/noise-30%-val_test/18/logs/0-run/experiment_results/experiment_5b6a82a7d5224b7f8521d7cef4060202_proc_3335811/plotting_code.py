import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- setup -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    datasets = list(experiment_data.keys())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data, datasets = {}, []


# ----------- helper -------------
def safe_get(d, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default


# ----------- per-dataset plots ---
for ds in datasets:
    ed = experiment_data[ds]
    epochs = ed.get("epochs", [])
    # 1) loss curves
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train")
        plt.plot(epochs, ed["losses"]["val"], linestyle="--", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds}: Training vs Validation Loss\nLeft: Train, Right: Val")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds}_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds}: {e}")
        plt.close()

    # 2) MCC curves
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["train"], label="Train")
        plt.plot(epochs, ed["metrics"]["val"], linestyle="--", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Matthews CorrCoef")
        plt.title(f"{ds}: Training vs Validation MCC\nLeft: Train, Right: Val")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds}_mcc_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC plot for {ds}: {e}")
        plt.close()

    # 3) confusion matrix on dev
    try:
        preds = np.array(ed.get("predictions", []))
        gts = np.array(ed.get("ground_truth", []))
        if preds.size and gts.size:
            cm = np.zeros((2, 2), dtype=int)
            for g, p in zip(gts, preds):
                cm[g, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.title(f"{ds} Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.colorbar()
            fname = os.path.join(working_dir, f"{ds}_confusion_matrix.png")
            plt.savefig(fname, dpi=150)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds}: {e}")
        plt.close()

# ----------- cross-dataset plot ---
if len(datasets) > 1:
    try:
        test_mccs = [
            safe_get(experiment_data[ds], "test_mcc", default=np.nan) for ds in datasets
        ]
        plt.figure()
        plt.bar(datasets, test_mccs, color="gray")
        plt.ylabel("Test MCC")
        plt.title("Comparison of Test MCC Across Datasets")
        for i, m in enumerate(test_mccs):
            plt.text(i, m, f"{m:.3f}", ha="center", va="bottom")
        fname = os.path.join(working_dir, "datasets_test_mcc_comparison.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating cross-dataset MCC plot: {e}")
        plt.close()

# ----------- print summary -------
for ds in datasets:
    print(f"{ds} test MCC: {experiment_data[ds].get('test_mcc')}")
