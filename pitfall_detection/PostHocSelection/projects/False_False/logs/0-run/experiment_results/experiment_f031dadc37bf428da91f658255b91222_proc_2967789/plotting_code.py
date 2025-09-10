import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load experiment data ------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]
    # ------------ extract arrays -----------------
    epochs = np.array(data.get("epochs", []))
    train_f1 = (
        np.array([t[0] for t in data["metrics"]["train"]])
        if data["metrics"]["train"]
        else np.array([])
    )
    val_f1 = (
        np.array([v[0] for v in data["metrics"]["val"]])
        if data["metrics"]["val"]
        else np.array([])
    )
    val_swa = (
        np.array([v[1] for v in data["metrics"]["val"]])
        if data["metrics"]["val"]
        else np.array([])
    )
    val_cwa = (
        np.array([v[2] for v in data["metrics"]["val"]])
        if data["metrics"]["val"]
        else np.array([])
    )
    val_scwa = (
        np.array([v[3] for v in data["metrics"]["val"]])
        if data["metrics"]["val"]
        else np.array([])
    )
    train_loss = (
        np.array(data["losses"]["train"]) if data["losses"]["train"] else np.array([])
    )
    preds = np.array(data.get("predictions", []))
    trues = np.array(data.get("ground_truth", []))

    # ----------------- Plot 1: F1 curves --------------------------
    try:
        plt.figure()
        if train_f1.size:
            plt.plot(epochs, train_f1, label="Train Macro-F1")
        if val_f1.size:
            plt.plot(epochs, val_f1, label="Val Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 over Epochs\nLeft: Train, Right: Val")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # ----------------- Plot 2: Loss curve -------------------------
    try:
        if train_loss.size:
            plt.figure()
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("SPR_BENCH Training Loss over Epochs")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_loss_curve.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # --------- Plot 3: Val SWA/CWA/SCWA curves -------------------
    try:
        if val_swa.size:
            plt.figure()
            plt.plot(epochs, val_swa, label="SWA")
            plt.plot(epochs, val_cwa, label="CWA")
            plt.plot(epochs, val_scwa, label="SCWA")
            plt.xlabel("Epoch")
            plt.ylabel("Weighted Accuracy")
            plt.title("SPR_BENCH Val SWA/CWA/SCWA over Epochs")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_weighted_acc_curve.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating weighted-acc curves: {e}")
        plt.close()

    # ---------------- Plot 4: Confusion Matrix -------------------
    try:
        from sklearn.metrics import confusion_matrix

        if preds.size and trues.size:
            cm = confusion_matrix(trues, preds)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("SPR_BENCH Confusion Matrix")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
