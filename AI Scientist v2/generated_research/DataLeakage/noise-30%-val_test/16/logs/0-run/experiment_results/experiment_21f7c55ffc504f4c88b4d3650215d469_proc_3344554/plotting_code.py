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
    ed = experiment_data["count_only_mlp"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    epochs = ed["epochs"]
    tr_loss = ed["losses"]["train"]
    val_loss = ed["losses"]["val"]
    tr_acc = [m["acc"] for m in ed["metrics"]["train"]]
    val_acc = [m["acc"] for m in ed["metrics"]["val"]]
    tr_mcc = [m["MCC"] for m in ed["metrics"]["train"]]
    val_mcc = [m["MCC"] for m in ed["metrics"]["val"]]
    tr_rma = [m["RMA"] for m in ed["metrics"]["train"]]
    val_rma = [m["RMA"] for m in ed["metrics"]["val"]]
    preds = np.array(ed.get("predictions", []))
    gts = np.array(ed.get("ground_truth", []))

    # ---------- 1. loss curve ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH – Loss Curve")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- 2. accuracy curve ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH – Accuracy Curve")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve: {e}")
        plt.close()

    # ---------- 3. MCC curve ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_mcc, label="Train")
        plt.plot(epochs, val_mcc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title("SPR_BENCH – MCC Curve")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_mcc_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curve: {e}")
        plt.close()

    # ---------- 4. RMA curve ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_rma, label="Train")
        plt.plot(epochs, val_rma, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Rule-Macro Acc")
        plt.title("SPR_BENCH – RMA Curve")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_rma_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating RMA curve: {e}")
        plt.close()

    # ---------- 5. confusion matrix ----------
    try:
        if preds.size and gts.size:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(gts, preds, labels=[0, 1])
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046)
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])
            plt.title("SPR_BENCH – Confusion Matrix")
            plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------- print test metrics ----------
    if "test_metrics" in ed:
        print("\nTest metrics:")
        for k, v in ed["test_metrics"].items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
