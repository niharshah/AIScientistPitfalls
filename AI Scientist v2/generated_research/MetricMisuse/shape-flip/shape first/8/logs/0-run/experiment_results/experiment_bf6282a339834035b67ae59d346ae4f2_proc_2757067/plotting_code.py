import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
experiment_path = os.path.join(working_dir, "experiment_data.npy")
try:
    experiment_data = np.load(experiment_path, allow_pickle=True).item()
    ed = experiment_data["color_blind"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

# Convenience
train_loss = ed["losses"]["train"]
val_loss = ed["losses"]["val"]
train_swa = ed["metrics"]["train"]
val_swa = ed["metrics"]["val"]
test_swa = ed["metrics"]["test_swa"]
preds = np.array(ed.get("predictions", []))
gtruth = np.array(ed.get("ground_truth", []))
epochs = np.arange(1, len(train_loss) + 1)

# ---------- plot 1: loss ----------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.title("SPR_BENCH Loss Curve\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- plot 2: accuracy ----------
try:
    plt.figure()
    plt.plot(epochs, train_swa, label="Train")
    plt.plot(epochs, val_swa, label="Validation")
    plt.title("SPR_BENCH Shape-Weighted Accuracy\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_swa_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# ---------- plot 3: test correctness histogram ----------
try:
    if preds.size and gtruth.size:
        correct = preds == gtruth
        counts = [np.sum(correct), np.sum(~correct)]
        plt.figure()
        plt.bar(["Correct", "Incorrect"], counts, color=["green", "red"])
        plt.title("SPR_BENCH Test Predictions\nCorrect vs Incorrect Counts")
        plt.ylabel("Count")
        fname = os.path.join(working_dir, "spr_bench_test_correctness.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating correctness plot: {e}")
    plt.close()

# ---------- print final metric ----------
print(f"Final Test Shape-Weighted Accuracy: {test_swa:.3f}")
