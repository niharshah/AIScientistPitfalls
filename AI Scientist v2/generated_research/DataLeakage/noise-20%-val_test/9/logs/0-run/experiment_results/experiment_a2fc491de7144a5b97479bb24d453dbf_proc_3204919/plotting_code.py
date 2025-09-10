import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = exp["Adam_Beta1"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

# helper: pick best beta by final val acc
best_beta, best_val = None, -1
for beta, store in runs.items():
    val_acc = store["metrics"]["val"][-1]
    if val_acc > best_val:
        best_beta, best_val = beta, val_acc

# ---------- 1/5 accuracy curves ----------
try:
    plt.figure()
    for beta, store in runs.items():
        epochs = range(1, len(store["metrics"]["train"]) + 1)
        plt.plot(epochs, store["metrics"]["train"], label=f"{beta} train")
        plt.plot(epochs, store["metrics"]["val"], ls="--", label=f"{beta} val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy Curves\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- 2/5 loss curves ----------
try:
    plt.figure()
    for beta, store in runs.items():
        epochs = range(1, len(store["losses"]["train"]) + 1)
        plt.plot(epochs, store["losses"]["train"], label=f"{beta} train")
        plt.plot(epochs, store["losses"]["val"], ls="--", label=f"{beta} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- 3/5 RBA curves ----------
try:
    plt.figure()
    for beta, store in runs.items():
        epochs = range(1, len(store["RBA"]) + 1)
        plt.plot(epochs, store["RBA"], label=f"{beta}")
    plt.xlabel("Epoch")
    plt.ylabel("RBA")
    plt.title("SPR_BENCH Rule-Based Accuracy (RBA) Across Epochs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_RBA_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RBA plot: {e}")
    plt.close()

# ---------- 4/5 final test accuracy bar ----------
try:
    plt.figure()
    betas = list(runs.keys())
    test_accs = [runs[b]["metrics"]["val"][-1] for b in betas]  # validation last
    plt.bar(betas, test_accs, color="skyblue")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Final Validation Accuracy per β₁")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_accuracy_bars.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# ---------- 5/5 confusion matrix for best beta ----------
try:
    from itertools import product

    store = runs[best_beta]
    preds = store["predictions"]
    gts = store["ground_truth"]
    classes = np.unique(gts)
    cm = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"SPR_BENCH Confusion Matrix – {best_beta} (best val acc)")
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix_best_beta.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
