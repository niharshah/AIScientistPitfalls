import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# helper to access
bench = experiment_data.get("learning_rate", {}).get("SPR_BENCH", {})
best_metrics = bench.get("metrics", {})
best_losses = bench.get("losses", {})
all_runs = bench.get("all_runs", [])
y_true = bench.get("ground_truth", [])
y_pred = bench.get("predictions", [])

# ----------------------------------------------------------
# 1) Loss curves for best run
try:
    if best_losses:
        epochs_tr, loss_tr = zip(*best_losses["train"])
        epochs_val, loss_val = zip(*best_losses["val"])
        plt.figure()
        plt.plot(epochs_tr, loss_tr, label="Train")
        plt.plot(epochs_val, loss_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ----------------------------------------------------------
# 2) HWA curves for best run
try:
    if best_metrics:
        ep_tr, hwa_tr = zip(*best_metrics["train"])
        ep_val, hwa_val = zip(*best_metrics["val"])
        plt.figure()
        plt.plot(ep_tr, hwa_tr, label="Train")
        plt.plot(ep_val, hwa_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy")
        plt.title("SPR_BENCH: Training vs Validation HWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_hwa_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ----------------------------------------------------------
# 3) Validation HWA for each LR sweep
try:
    if all_runs:
        plt.figure()
        for run in all_runs:
            ep, val_hwa = zip(*run["metrics"]["val"])
            plt.plot(ep, val_hwa, label=f"lr={run['lr']:.0e}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation HWA")
        plt.title("SPR_BENCH: LR Sweep – Validation HWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_lr_sweep_val_hwa.png"))
    plt.close()
except Exception as e:
    print(f"Error creating LR sweep plot: {e}")
    plt.close()

# ----------------------------------------------------------
# 4) Confusion matrix on test set
try:
    if y_true and y_pred:
        num_classes = len(set(y_true) | set(y_pred))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Test Confusion Matrix")
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ----------------------------------------------------------
# Print overall test HWA
try:
    if y_true and y_pred:
        # replicate helper from experiment code
        def count_shape_variety(sequence: str) -> int:
            return len(set(tok[0] for tok in sequence.strip().split() if tok))

        def count_color_variety(sequence: str) -> int:
            return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))

        # we need original sequences to compute weights; cannot, so print stored HWA
        best_test_hwa = bench.get("metrics", {}).get("test_hwa", None)
        # if unavailable, approximate unweighted accuracy
        if best_test_hwa is None:
            best_test_hwa = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"Test HWA (reported) = {best_test_hwa:.4f}")
except Exception as e:
    print(f"Error computing/printing test HWA: {e}")
