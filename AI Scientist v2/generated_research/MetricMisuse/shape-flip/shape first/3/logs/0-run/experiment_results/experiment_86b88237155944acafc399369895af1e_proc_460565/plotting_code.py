import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

run = experiment_data.get("No-PE", {}).get("SPR_BENCH", {})
losses = run.get("losses", {})
metrics = run.get("metrics", {})
preds = run.get("predictions", [])
truths = run.get("ground_truth", [])

# -------------------------------------------------------------------
# 1) Loss curves
try:
    tr_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    if tr_loss and val_loss:
        epochs = np.arange(1, len(tr_loss) + 1)
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Task Loss")
        plt.title("SPR_BENCH No-PE: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_NoPE_loss_curve.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# -------------------------------------------------------------------
# 2) Validation Shape-Weighted Accuracy
try:
    val_swa = metrics.get("val", [])
    if val_swa:
        epochs = np.arange(1, len(val_swa) + 1)
        plt.figure()
        plt.plot(epochs, val_swa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH No-PE: Validation SWA over Epochs")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_NoPE_SWA_curve.png"))
        plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# -------------------------------------------------------------------
# 3) Confusion matrix on test set
try:
    if preds and truths:
        labels = sorted(set(truths) | set(preds))
        label2idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(truths, preds):
            cm[label2idx[t], label2idx[p]] += 1

        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH No-PE: Test Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_NoPE_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
