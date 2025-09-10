import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_Hybrid" in experiment_data:
    data = experiment_data["SPR_Hybrid"]

    # --------- unpack losses and metrics ----------
    ep_tr_loss = [e for e, _ in data["losses"]["train"]]
    tr_loss = [v for _, v in data["losses"]["train"]]
    val_loss = [v for _, v in data["losses"]["val"]]

    tr_swa = [v for _, v in data["metrics"]["train_SWA"]]
    val_swa = [v for _, v in data["metrics"]["val_SWA"]]

    gts = np.asarray(data["ground_truth"])
    preds = np.asarray(data["predictions"])
    test_acc = (gts == preds).mean() if len(gts) else np.nan
else:
    print("No SPR_Hybrid results found.")
    test_acc = np.nan

# -------------------------------------------------------------- #
# Plot 1 : Loss curves
try:
    plt.figure()
    plt.plot(ep_tr_loss, tr_loss, label="Train")
    plt.plot(ep_tr_loss, val_loss, label="Validation")
    plt.title("SPR_Hybrid – Cross-Entropy Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_Hybrid_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# -------------------------------------------------------------- #
# Plot 2 : Shape-Weighted Accuracy curves
try:
    plt.figure()
    plt.plot(ep_tr_loss, tr_swa, label="Train SWA")
    plt.plot(ep_tr_loss, val_swa, label="Validation SWA")
    plt.title("SPR_Hybrid – Shape-Weighted Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_Hybrid_SWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA curves: {e}")
    plt.close()

# -------------------------------------------------------------- #
# Plot 3 : Confusion matrix
try:
    if len(gts):
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(gts, preds):
            cm[int(t), int(p)] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.colorbar()
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title("SPR_Hybrid – Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, "SPR_Hybrid_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------------------------------------------------------------- #
# Print evaluation metric
print(f"SPR_Hybrid Test Accuracy: {test_acc:.4f}")
