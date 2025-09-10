import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- paths -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ----------------- per-dataset curves -----------------
for ds_name, rec in experiment_data.items():
    try:
        epochs = rec["epochs"]
        tr_loss, va_loss = rec["losses"]["train"], rec["losses"]["val"]
        tr_f1, va_f1 = rec["metrics"]["train_f1"], rec["metrics"]["val_f1"]

        plt.figure(figsize=(10, 4))
        # Left subplot : Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, va_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()

        # Right subplot : F1
        plt.subplot(1, 2, 2)
        plt.plot(epochs, tr_f1, label="Train")
        plt.plot(epochs, va_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("Macro F1")
        plt.legend()

        plt.suptitle(f"{ds_name} | Left: Loss Curve, Right: F1 Curve")
        fname = os.path.join(working_dir, f"{ds_name}_loss_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {ds_name}: {e}")
        plt.close()

# ----------------- aggregate bar chart -----------------
try:
    datasets = list(experiment_data.keys())
    test_f1s = [experiment_data[d]["test_f1"] for d in datasets]

    plt.figure()
    plt.bar(datasets, test_f1s, color="skyblue")
    plt.ylabel("Test Macro F1")
    plt.title("Test Performance Across Datasets")
    fname = os.path.join(working_dir, "all_datasets_test_macroF1.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating aggregate plot: {e}")
    plt.close()

# ----------------- print test metrics -----------------
for ds_name, rec in experiment_data.items():
    print(
        f"[{ds_name}] Test -> loss: {rec['test_loss']:.4f}, macroF1: {rec['test_f1']:.4f}, EMA: {rec['test_ema']:.4f}"
    )
