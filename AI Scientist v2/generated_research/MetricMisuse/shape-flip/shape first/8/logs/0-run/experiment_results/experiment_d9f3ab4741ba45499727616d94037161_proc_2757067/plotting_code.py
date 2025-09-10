import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helpers ----------
def get_spr():
    algo = "NoSymbolicFallback"
    dataset = "spr_bench"
    return experiment_data.get(algo, {}).get(dataset, {})


spr = get_spr()
if not spr:
    exit()

epochs = list(range(1, len(spr["metrics"]["train_swa"]) + 1))

# ---------- plot 1: SWA curve ----------
try:
    plt.figure()
    plt.plot(epochs, spr["metrics"]["train_swa"], label="Train SWA")
    plt.plot(epochs, spr["metrics"]["val_swa"], label="Validation SWA")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("spr_bench: Training vs Validation SWA")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_swa_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# ---------- plot 2: loss curve ----------
try:
    plt.figure()
    plt.plot(epochs, spr["losses"]["train"], label="Train Loss")
    plt.plot(epochs, spr["losses"]["val"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("spr_bench: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- plot 3: confusion matrix ----------
try:
    preds = np.array(spr["predictions"])
    gts = np.array(spr["ground_truth"])
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("spr_bench: Confusion Matrix (Test Set)")
    fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- print evaluation metric ----------
print(f"Test Shape-Weighted Accuracy (SWA): {spr['metrics']['test_swa']:.3f}")
