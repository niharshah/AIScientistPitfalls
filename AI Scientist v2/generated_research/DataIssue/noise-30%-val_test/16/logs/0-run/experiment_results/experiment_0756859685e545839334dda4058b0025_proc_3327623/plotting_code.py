import matplotlib.pyplot as plt
import numpy as np
import os

# --- setup -------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    data = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

epochs = data["epochs"]
train_loss = data["losses"]["train"]
val_loss = data["losses"]["val"]
train_mcc = data["metrics"]["train_MCC"]
val_mcc = data["metrics"]["val_MCC"]
test_pred = np.array(data["predictions"])
test_true = np.array(data["ground_truth"])

# 1) loss curve ---------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("SPR_BENCH – Loss Curve")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) MCC curve ----------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_mcc, label="Train MCC")
    plt.plot(epochs, val_mcc, label="Validation MCC")
    plt.xlabel("Epoch")
    plt.ylabel("Matthews Corr. Coef.")
    plt.title("SPR_BENCH – MCC Curve")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_MCC_curve.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating MCC curve: {e}")
    plt.close()

# 3) Test confusion matrix (bar) ---------------------------------------------
try:
    tp = int(((test_pred == 1) & (test_true == 1)).sum())
    tn = int(((test_pred == 0) & (test_true == 0)).sum())
    fp = int(((test_pred == 1) & (test_true == 0)).sum())
    fn = int(((test_pred == 0) & (test_true == 1)).sum())
    counts = [tp, tn, fp, fn]
    labels = ["TP", "TN", "FP", "FN"]
    plt.figure()
    plt.bar(labels, counts, color=["green", "blue", "orange", "red"])
    for idx, c in enumerate(counts):
        plt.text(idx, c + 0.5, str(c), ha="center")
    plt.ylabel("Count")
    plt.title("SPR_BENCH – Test Confusion Matrix")
    save_path = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# quick console confirmation
print("Plotted epochs:", epochs)
print("Train loss:", train_loss)
print("Val loss:", val_loss)
print("Train MCC:", train_mcc)
print("Val MCC:", val_mcc)
