import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- Load data -------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

exp = experiment_data["CLS_AGG"]["SPR_BENCH"]
loss_tr = np.array(exp["losses"]["train"])
loss_val = np.array(exp["losses"]["val"])
met_tr = np.array(exp["metrics"]["train"])
met_val = np.array(exp["metrics"]["val"])
gts = np.array(exp["ground_truth"])
preds = np.array(exp["predictions"])

# ------------- Figure 1: Loss curves -------------
try:
    plt.figure()
    plt.plot(loss_tr[:, 0], loss_tr[:, 1], label="Train")
    plt.plot(loss_val[:, 0], loss_val[:, 1], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss vs Epochs\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------- Figure 2: SWA curves -------------
try:
    plt.figure()
    plt.plot(met_tr[:, 0], met_tr[:, 1], label="Train")
    plt.plot(met_val[:, 0], met_val[:, 1], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR_BENCH SWA vs Epochs\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_SWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# ------------- Figure 3: Confusion matrix -------------
try:
    from itertools import product

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i, j in product(range(2), repeat=2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------- Compute & print test accuracy -------------
test_acc = (gts == preds).mean() if len(gts) else float("nan")
print(f"Test Accuracy: {test_acc:.4f}")
