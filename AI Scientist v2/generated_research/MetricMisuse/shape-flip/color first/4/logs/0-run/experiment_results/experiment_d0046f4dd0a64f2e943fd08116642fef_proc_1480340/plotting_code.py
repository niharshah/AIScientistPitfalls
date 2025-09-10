import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp = experiment_data.get("SPR_BENCH", {})

# ----------------- Plot 1: Train vs Val loss ----------------------
try:
    train_loss = exp["metrics"]["train_loss"]
    val_loss = exp["metrics"]["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train Loss, Right: Validation Loss")
    plt.legend()
    fname = "SPR_BENCH_loss_curve.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ----------------- Plot 2: CWA curve ------------------------------
try:
    cwa = exp["metrics"]["CompWA"]
    epochs = range(1, len(cwa) + 1)

    plt.figure()
    plt.plot(epochs, cwa, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.title("SPR_BENCH Validation CWA Over Epochs")
    fname = "SPR_BENCH_CWA_curve.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating CWA curve: {e}")
    plt.close()

# ----------------- Plot 3: Confusion Matrix -----------------------
try:
    preds = np.array(exp["predictions"])
    gts = np.array(exp["ground_truth"])
    labels = sorted(set(gts) | set(preds))
    n_cls = len(labels)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(gts, preds):
        cm[labels.index(t), labels.index(p)] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xticks(range(n_cls), labels)
    plt.yticks(range(n_cls), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("SPR_BENCH Confusion Matrix\nGround Truth vs Predicted Labels")
    fname = "SPR_BENCH_confusion_matrix.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
