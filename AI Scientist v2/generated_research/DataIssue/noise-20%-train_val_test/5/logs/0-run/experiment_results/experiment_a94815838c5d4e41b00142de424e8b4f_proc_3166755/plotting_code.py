import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------- setup & data loading ------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

test_acc_dict = {}  # to collect test accuracies for bar plot

# ------------------ per-dataset visualisations -------------------- #
for dname, ddata in experiment_data.items():
    # --- compute test accuracy from predictions & gts ------------- #
    preds = np.array(ddata.get("predictions", []))
    gts = np.array(ddata.get("ground_truth", []))
    test_acc = (preds == gts).mean() if len(preds) else np.nan
    test_acc_dict[dname] = test_acc

    epochs = np.arange(1, len(ddata["metrics"]["train_acc"]) + 1)

    # 1) accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, ddata["metrics"]["train_acc"], marker="o", label="train")
        plt.plot(epochs, ddata["metrics"]["val_acc"], marker="x", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dname} Accuracy Curves")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_accuracy_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dname}: {e}")
        plt.close()

    # 2) loss curves
    try:
        plt.figure()
        plt.plot(epochs, ddata["losses"]["train"], marker="o", label="train")
        plt.plot(epochs, ddata["losses"]["val"], marker="x", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname} Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # 3) confusion matrix
    try:
        if len(preds) and len(gts):
            num_classes = int(max(gts.max(), preds.max()) + 1)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for p, g in zip(preds, gts):
                cm[g, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dname} Confusion Matrix (Test)")
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

# ---------------- global comparison plot -------------------------- #
try:
    plt.figure()
    names = list(test_acc_dict.keys())
    values = [test_acc_dict[n] for n in names]
    plt.bar(names, values, color="skyblue")
    plt.ylabel("Test Accuracy")
    plt.title("Dataset Test Accuracy Comparison")
    fname = os.path.join(working_dir, "test_accuracy_comparison.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating comparison bar chart: {e}")
    plt.close()

print("Test accuracies:", test_acc_dict)
