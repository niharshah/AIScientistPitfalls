import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------ SETUP ------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ LOAD EXPERIMENT DATA ---------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------ INIT GLOBAL LISTS ------------------------------
all_dnames, all_test_acc, all_irf = [], [], []

# ------------------ PER-DATASET PLOTS ------------------------------
for dname, ddict in experiment_data.items():
    losses = ddict.get("losses", {})
    metrics = ddict.get("metrics", {})
    y_pred = np.array(ddict.get("predictions", []))
    y_true = np.array(ddict.get("ground_truth", []))

    # Collect summary stats for global comparison
    test_acc = float(metrics.get("test", [np.nan])[0])
    irf_val = float(metrics.get("irf", [np.nan])[0])
    all_dnames.append(dname)
    all_test_acc.append(test_acc)
    all_irf.append(irf_val)

    # 1) Validation loss curve
    try:
        val_loss = losses.get("val", [])
        if len(val_loss):
            plt.figure()
            plt.plot(range(1, len(val_loss) + 1), val_loss, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.title(f"{dname} – Validation Loss")
            fname = os.path.join(working_dir, f"{dname}_val_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating val loss plot for {dname}: {e}")
        plt.close()

    # 2) Validation accuracy curve
    try:
        val_acc = metrics.get("val", [])
        if len(val_acc):
            plt.figure()
            plt.plot(range(1, len(val_acc) + 1), val_acc, marker="o", color="green")
            plt.xlabel("Epoch")
            plt.ylabel("Validation Accuracy")
            plt.title(f"{dname} – Validation Accuracy")
            fname = os.path.join(working_dir, f"{dname}_val_accuracy.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating val accuracy plot for {dname}: {e}")
        plt.close()

    # 3) Confusion matrix
    try:
        if y_true.size and y_pred.size:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_true, y_pred)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.title(f"{dname} – Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for (i, j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha="center", va="center")
            plt.colorbar(im)
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

# ------------------ GLOBAL COMPARISON PLOTS ------------------------
try:
    if all_dnames:
        # 4) Test accuracy comparison
        plt.figure()
        plt.bar(all_dnames, all_test_acc, color="orange")
        plt.ylabel("Test Accuracy")
        plt.title("Dataset Comparison – Test Accuracy")
        fname = os.path.join(working_dir, "comparison_test_accuracy.png")
        plt.savefig(fname)
        plt.close()

        # 5) IRF comparison
        plt.figure()
        plt.bar(all_dnames, all_irf, color="purple")
        plt.ylabel("IRF")
        plt.title("Dataset Comparison – IRF")
        fname = os.path.join(working_dir, "comparison_irf.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating comparison plots: {e}")
    plt.close()

# ------------------ PRINT SUMMARY ----------------------------------
for d, acc, irf in zip(all_dnames, all_test_acc, all_irf):
    print(f"{d}: test_acc={acc:.4f}, irf={irf:.4f}")
