import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    lr_dict = experiment_data.get("learning_rate", {})
    lrs = sorted(lr_dict.keys(), key=lambda x: float(x.split("=")[1]))
    # prepare containers
    loss_tr, loss_val, acc_tr, acc_val, acc_test = {}, {}, {}, {}, {}
    epochs = None
    for k in lrs:
        entry = lr_dict[k]["SPR"]
        loss_tr[k] = entry["losses"]["train"]
        loss_val[k] = entry["losses"]["val"]
        acc_tr[k] = entry["metrics"]["train"]
        acc_val[k] = entry["metrics"]["val"]
        acc_test[k] = entry["metrics"]["test"]
        epochs = entry["epochs"]  # same for all

    # ---------- plot losses ----------
    try:
        plt.figure()
        for k in lrs:
            plt.plot(epochs, loss_tr[k], "--", label=f"{k} train")
            plt.plot(epochs, loss_val[k], "-", label=f"{k} val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Training vs Validation Loss across Learning Rates")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_loss_curves_all_lrs.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- plot accuracies ----------
    try:
        plt.figure()
        for k in lrs:
            plt.plot(epochs, acc_tr[k], "--", label=f"{k} train")
            plt.plot(epochs, acc_val[k], "-", label=f"{k} val")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title("Training vs Validation CpxWA across Learning Rates")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_accuracy_curves_all_lrs.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------- bar chart of test accuracy ----------
    try:
        plt.figure()
        labels = [k for k in lrs]
        vals = [acc_test[k] for k in lrs]
        plt.bar(labels, vals, color="skyblue")
        plt.ylabel("Test Complexity-Weighted Accuracy")
        plt.title("Test CpxWA vs Learning Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_test_accuracy_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy bar plot: {e}")
        plt.close()

    # --------- print evaluation metrics ----------
    print("Final Test Complexity-Weighted Accuracies:")
    for k in lrs:
        print(f"{k}: {acc_test[k]:.4f}")
