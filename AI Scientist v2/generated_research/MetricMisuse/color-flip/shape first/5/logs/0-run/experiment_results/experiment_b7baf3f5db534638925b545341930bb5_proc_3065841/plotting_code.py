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

for dname, data in experiment_data.items():
    epochs = data.get("epochs", [])
    tr_loss = data.get("losses", {}).get("train", [])
    val_loss = data.get("losses", {}).get("val", [])
    tr_hsca = data.get("metrics", {}).get("train_HSCA", [])
    val_hsca = data.get("metrics", {}).get("val_HSCA", [])
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname} – Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # 2) HSCA curves
    try:
        plt.figure()
        plt.plot(epochs, tr_hsca, label="Train HSCA")
        plt.plot(epochs, val_hsca, label="Validation HSCA")
        plt.xlabel("Epoch")
        plt.ylabel("HSCA")
        plt.title(f"{dname} – Training vs Validation HSCA")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_hsca_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HSCA plot for {dname}: {e}")
        plt.close()

    # 3) Confusion-matrix-like heat-map (only if labels available)
    if preds.size and gts.size:
        try:
            num_classes = int(max(preds.max(), gts.max())) + 1
            conf = np.zeros((num_classes, num_classes), dtype=int)
            for gt, pr in zip(gts, preds):
                conf[gt, pr] += 1
            plt.figure()
            plt.imshow(conf, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dname} – Confusion Matrix (Last Epoch)")
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {dname}: {e}")
            plt.close()

    # -------- print evaluation metrics --------
    final_hsca = val_hsca[-1] if val_hsca else float("nan")
    best_hsca = max(val_hsca) if val_hsca else float("nan")
    accuracy = (preds == gts).mean() if preds.size else float("nan")
    print(
        f"{dname}: final HSCA={final_hsca:.4f}, best HSCA={best_hsca:.4f}, accuracy={accuracy:.4f}"
    )
