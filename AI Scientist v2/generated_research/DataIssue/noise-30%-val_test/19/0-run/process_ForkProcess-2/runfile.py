import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------- plotting ------------------
for dset, info in experiment_data.items():
    epochs = info.get("epochs", [])
    losses_tr = info.get("losses", {}).get("train", [])
    losses_val = info.get("losses", {}).get("val", [])
    f1_tr = info.get("metrics", {}).get("train_macro_f1", [])
    f1_val = info.get("metrics", {}).get("val_macro_f1", [])
    y_true = np.array(info.get("ground_truth", []))
    y_pred = np.array(info.get("predictions", []))

    # 1. Loss curve
    try:
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train")
        plt.plot(epochs, losses_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset}: Train vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dset}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dset}: {e}")
        plt.close()

    # 2. Macro-F1 curve
    try:
        plt.figure()
        plt.plot(epochs, f1_tr, label="Train")
        plt.plot(epochs, f1_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dset}: Train vs Validation Macro-F1")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dset}_macro_f1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve for {dset}: {e}")
        plt.close()

    # 3. Confusion matrix (only if preds exist)
    try:
        if y_true.size and y_pred.size:
            cm = np.zeros((2, 2), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[int(t), int(p)] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])
            plt.title(f"{dset}: Confusion Matrix (Test Set)")
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()
