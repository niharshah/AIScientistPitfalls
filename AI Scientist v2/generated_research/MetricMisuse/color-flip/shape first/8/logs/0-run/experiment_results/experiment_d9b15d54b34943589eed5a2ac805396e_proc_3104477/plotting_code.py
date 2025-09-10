import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    ed = experiment_data["CLS_POOL_ONLY"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

epochs = np.arange(1, len(ed["losses"]["train"]) + 1)

# 1) Loss curves -------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, ed["losses"]["train"], label="Train Loss")
    plt.plot(epochs, ed["losses"]["val"], label="Val Loss")
    plt.title("SPR_BENCH – Training vs Validation Loss\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) CCWA metric -------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, ed["metrics"]["val_CCWA"], marker="o")
    plt.title("SPR_BENCH – Validation CCWA over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CCWA")
    plt.ylim(0, 1)
    save_path = os.path.join(working_dir, "SPR_BENCH_val_CCWA.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating CCWA plot: {e}")
    plt.close()

# 3) Confusion matrix (final epoch) ------------------------------
try:
    preds = np.array(ed["predictions"][-1])
    labels = np.array(ed["ground_truth"][-1])
    n_cls = int(max(labels.max(), preds.max()) + 1)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.title("SPR_BENCH – Confusion Matrix (Final Epoch)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    # annotate
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
            )
    save_path = os.path.join(working_dir, "SPR_BENCH_confusion_matrix_final.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

print("All figures saved to", working_dir)
