import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
models = list(experiment_data.keys())
val_accs, val_losses = [], []

for m in models:
    try:
        val_accs.append(experiment_data[m][dataset]["metrics"]["val"][0])
        val_losses.append(experiment_data[m][dataset]["losses"]["val"][0])
    except Exception:
        val_accs.append(np.nan)
        val_losses.append(np.nan)

print("Validation accuracies:", dict(zip(models, val_accs)))
print("Validation losses:", dict(zip(models, val_losses)))

# ---------- plot 1: validation accuracy ----------
try:
    plt.figure()
    plt.bar(models, val_accs, color="skyblue")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset} Validation Accuracy per Model")
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset}_val_accuracy.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- plot 2: validation loss ----------
try:
    plt.figure()
    plt.bar(models, val_losses, color="salmon")
    plt.ylabel("Log Loss")
    plt.title(f"{dataset} Validation Loss per Model")
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset}_val_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- plot 3: label distribution ----------
try:
    plt.figure()
    width = 0.35
    x = np.arange(2)  # labels 0 and 1
    # Use the last model (if multiple) for predictions plot
    gt = experiment_data[models[-1]][dataset]["ground_truth"]
    pred = experiment_data[models[-1]][dataset]["predictions"]
    gt_cnt = np.bincount(gt, minlength=2)
    pr_cnt = np.bincount(pred, minlength=2)
    plt.bar(x - width / 2, gt_cnt, width, label="Ground Truth")
    plt.bar(x + width / 2, pr_cnt, width, label="Predictions")
    plt.xticks(x, ["Label 0", "Label 1"])
    plt.ylabel("Count")
    plt.title(f"Label Distribution — {dataset}\nLeft: Ground Truth, Right: Predictions")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, f"{dataset}_label_distribution.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating label distribution plot: {e}")
    plt.close()
