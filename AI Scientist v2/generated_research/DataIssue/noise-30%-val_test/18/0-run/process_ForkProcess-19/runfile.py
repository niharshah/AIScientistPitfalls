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
    experiment_data = None

if experiment_data is not None:
    run = experiment_data["SingleHeadAttention"]["SPR_BENCH"]
    tr_loss, va_loss = run["losses"]["train"], run["losses"]["val"]
    tr_mcc, va_mcc = run["metrics"]["train_MCC"], run["metrics"]["val_MCC"]
    preds, gts = run["predictions"], run["ground_truth"]

    # print final evaluation metrics
    try:
        from sklearn.metrics import matthews_corrcoef, f1_score

        test_mcc = matthews_corrcoef(gts, preds) if preds else float("nan")
        test_f1 = f1_score(gts, preds, average="macro") if preds else float("nan")
        print(f"Test MCC={test_mcc:.4f} | Test F1={test_f1:.4f}")
    except Exception as e:
        print(f"Error computing metrics: {e}")

    # -------- plot 1: loss curves --------
    try:
        plt.figure()
        plt.plot(tr_loss, label="Train Loss")
        plt.plot(va_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------- plot 2: MCC curves --------
    try:
        plt.figure()
        plt.plot(tr_mcc, label="Train MCC")
        plt.plot(va_mcc, label="Val MCC")
        plt.xlabel("Epoch")
        plt.ylabel("Matthews Corr. Coef.")
        plt.title("SPR_BENCH: Training vs Validation MCC")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_mcc_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating MCC plot: {e}")
        plt.close()

    # -------- plot 3: confusion matrix --------
    try:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("SPR_BENCH Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
