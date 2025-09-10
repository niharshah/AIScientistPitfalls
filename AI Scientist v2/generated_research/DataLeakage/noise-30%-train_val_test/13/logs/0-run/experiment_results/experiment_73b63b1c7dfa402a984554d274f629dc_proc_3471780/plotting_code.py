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

# unify single-run vs multi-run access
spr_raw = experiment_data.get("SPR_BENCH", {})
if not spr_raw:
    print("No SPR_BENCH data found.")
    exit()

# put everything into a dict of runs
if "metrics" in spr_raw:  # single run
    runs = {"run_1": spr_raw}
else:  # already multiple runs
    runs = spr_raw


# helper
def arr(run_key, section, name):
    return np.asarray(runs[run_key][section][name])


# ---------- plot 1: F1 curves ----------
try:
    plt.figure()
    for k in runs:
        plt.plot(runs[k]["epochs"], arr(k, "metrics", "train_f1"), label=f"{k} train")
        plt.plot(
            runs[k]["epochs"],
            arr(k, "metrics", "val_f1"),
            linestyle="--",
            label=f"{k} val",
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
    for k in runs:
        plt.plot(runs[k]["epochs"], arr(k, "losses", "train"), label=f"{k} train")
        plt.plot(
            runs[k]["epochs"], arr(k, "losses", "val"), linestyle="--", label=f"{k} val"
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
for k in runs:
    preds = np.asarray(runs[k]["predictions"])
    gts = np.asarray(runs[k]["ground_truth"])
    if preds.size and gts.size:
        test_f1s[k] = f1_score(gts, preds, average="macro")

# ---------- plot 3: bar chart of test F1 ----------
if len(test_f1s) > 1:
    try:
        plt.figure()
        keys = list(test_f1s.keys())
        vals = [test_f1s[k] for k in keys]
        plt.bar(keys, vals, color="skyblue")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Test Macro-F1 by Run")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating bar plot: {e}")
        plt.close()

# ---------- plot 4: confusion matrix for best run ----------
try:
    best_run = max(test_f1s, key=test_f1s.get)
    preds = np.asarray(runs[best_run]["predictions"])
    gts = np.asarray(runs[best_run]["ground_truth"])
    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title(f"SPR_BENCH Confusion Matrix ({best_run})")
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
    plt.savefig(os.path.join(working_dir, f"SPR_BENCH_confusion_{best_run}.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- print summary ----------
print("Test Macro-F1 scores:")
for k, v in test_f1s.items():
    print(f"{k}: {v:.4f}")
