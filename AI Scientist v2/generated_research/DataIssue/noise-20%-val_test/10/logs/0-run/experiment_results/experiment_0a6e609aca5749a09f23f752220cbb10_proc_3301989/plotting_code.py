import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("hidden_dim", {}).get("SPR_BENCH", {})
hidden_dims = sorted([int(k) for k in runs.keys()])
best_dev_f1, test_f1s, dev_f1s = [], [], []
for hd in hidden_dims:
    r = runs[str(hd)]
    dev_f1s.append(max(r["metrics"]["val_f1"]))
    test_f1s.append(r["metrics"]["test_f1"][0])
    best_dev_f1.append(max(r["metrics"]["val_f1"]))

# ---------------- plot 1: hidden-dim tuning -----------------
try:
    plt.figure()
    plt.plot(hidden_dims, dev_f1s, marker="o")
    plt.xlabel("Hidden Dim")
    plt.ylabel("Best Dev Macro-F1")
    plt.title("SPR_BENCH: Hidden Dim Tuning (Dev Macro-F1)")
    fname = os.path.join(working_dir, "SPR_BENCH_hidden_dim_tuning_dev_macroF1.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating hidden-dim tuning plot: {e}")
    plt.close()

# ---------------- plot 2: f1 curves -----------------
try:
    plt.figure()
    for hd in hidden_dims:
        r = runs[str(hd)]
        plt.plot(r["metrics"]["train_f1"], label=f"train hd={hd}", linestyle="--")
        plt.plot(r["metrics"]["val_f1"], label=f"val hd={hd}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Train vs Val Macro-F1 Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# ---------------- plot 3: loss curves -----------------
try:
    plt.figure()
    for hd in hidden_dims:
        r = runs[str(hd)]
        plt.plot(r["losses"]["train"], label=f"train hd={hd}", linestyle="--")
        plt.plot(r["losses"]["val"], label=f"val hd={hd}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Train vs Val Loss Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------------- plot 4: confusion matrix for best model -----------------
try:
    # pick hidden_dim with best test f1
    best_idx = int(np.argmax(test_f1s))
    best_hd = hidden_dims[best_idx]
    preds = np.array(runs[str(best_hd)]["predictions"])
    gts = np.array(runs[str(best_hd)]["ground_truth"])
    classes = np.unique(np.concatenate([preds, gts]))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"SPR_BENCH: Confusion Matrix (best hd={best_hd})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
            )
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------------- print brief metrics table -----------------
print("HiddenDim | BestDevF1 | TestF1")
for hd, d, t in zip(hidden_dims, dev_f1s, test_f1s):
    print(f"{hd:9} | {d:.4f}    | {t:.4f}")
