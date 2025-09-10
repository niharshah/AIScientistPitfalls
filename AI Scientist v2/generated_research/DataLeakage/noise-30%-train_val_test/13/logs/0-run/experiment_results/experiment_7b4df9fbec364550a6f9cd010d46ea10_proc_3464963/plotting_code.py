import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix

# ---------- load ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

runs = experiment_data.get("num_layers", {}).get("SPR_BENCH", {})
if not runs:
    print("No SPR_BENCH data found.")
    exit()


# helper: fetch arrays
def extract(metric_type, split, nl_key):
    return np.asarray(runs[nl_key][metric_type][split])


# ---------- plot 1: F1 curves ----------
try:
    plt.figure()
    for nl_key in sorted(runs, key=lambda x: int(x.split("_")[1])):
        epochs = runs[nl_key]["epochs"]
        plt.plot(
            epochs, extract("metrics", "train_f1", nl_key), label=f"train nl={nl_key}"
        )
        plt.plot(
            epochs,
            extract("metrics", "val_f1", nl_key),
            linestyle="--",
            label=f"val nl={nl_key}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Training vs Validation Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# ---------- plot 2: Loss curves ----------
try:
    plt.figure()
    for nl_key in sorted(runs, key=lambda x: int(x.split("_")[1])):
        epochs = runs[nl_key]["epochs"]
        plt.plot(epochs, extract("losses", "train", nl_key), label=f"train nl={nl_key}")
        plt.plot(
            epochs,
            extract("losses", "val", nl_key),
            linestyle="--",
            label=f"val nl={nl_key}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# compute test F1s
test_f1s = {}
for nl_key in runs:
    preds = np.asarray(runs[nl_key]["predictions"])
    gts = np.asarray(runs[nl_key]["ground_truth"])
    test_f1s[nl_key] = f1_score(gts, preds, average="macro")

# ---------- plot 3: bar chart of test F1 ----------
try:
    plt.figure()
    keys = sorted(test_f1s, key=lambda x: int(x.split("_")[1]))
    vals = [test_f1s[k] for k in keys]
    plt.bar(keys, vals, color="skyblue")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Test Macro-F1 by num_layers")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# identify best model
best_key = max(test_f1s, key=test_f1s.get)

# ---------- plot 4: confusion matrix for best model ----------
try:
    preds = np.asarray(runs[best_key]["predictions"])
    gts = np.asarray(runs[best_key]["ground_truth"])
    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title(f"SPR_BENCH Confusion Matrix (Best {best_key})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"SPR_BENCH_confusion_{best_key}.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- print summary ----------
print("Test Macro-F1 scores:")
for k, v in sorted(test_f1s.items(), key=lambda x: int(x[0].split("_")[1])):
    print(f"{k}: {v:.4f}")
