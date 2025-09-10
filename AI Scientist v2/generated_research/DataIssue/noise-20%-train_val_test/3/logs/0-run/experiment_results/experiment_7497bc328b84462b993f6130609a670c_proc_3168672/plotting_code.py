import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["No_Label_Smoothing"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

epochs = ed["epochs"]
tr_loss, val_loss = ed["losses"]["train"], ed["losses"]["val"]
tr_f1, val_f1 = ed["metrics"]["train"], ed["metrics"]["val"]
preds, gts = ed["predictions"], ed["ground_truth"]

# -------------------- plot 1: loss curve --------------------
try:
    plt.figure()
    plt.plot(epochs, tr_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curve\nTrain vs. Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_Loss_Curve_No_Label_Smoothing.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# -------------------- plot 2: F1 curve --------------------
try:
    plt.figure()
    plt.plot(epochs, tr_f1, label="Train")
    plt.plot(epochs, val_f1, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 Curve\nTrain vs. Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_F1_Curve_No_Label_Smoothing.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# -------------------- plot 3: confusion matrix --------------------
try:
    cm = confusion_matrix(gts, preds)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("SPR_BENCH Confusion Matrix (Test Set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fname = os.path.join(
        working_dir, "SPR_BENCH_Confusion_Matrix_No_Label_Smoothing.png"
    )
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------------------- evaluation metric --------------------
try:
    test_f1 = f1_score(gts, preds, average="macro")
    print(f"Final Test Macro-F1: {test_f1:.4f}")
except Exception as e:
    print(f"Error computing final F1: {e}")
