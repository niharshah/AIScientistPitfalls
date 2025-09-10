import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def macro_f1(pred, true, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1s.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


# ------------------------------------------------------------------ #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    raise RuntimeError(f"Could not load or locate experiment data: {e}")

metrics, losses = spr["metrics"], spr["losses"]
preds, gts = np.array(spr["predictions"]), np.array(spr["ground_truth"])
n_cls = len(np.unique(gts))

# ---------------- PLOTTING SECTION --------------------------------- #
plots = [
    ("loss_curve", losses["train"], losses["val"], "Loss", "Train vs. Val Loss"),
    (
        "accuracy_curve",
        metrics["train_acc"],
        metrics["val_acc"],
        "Accuracy",
        "Train vs. Val Accuracy",
    ),
    (
        "f1_curve",
        metrics["train_f1"],
        metrics["val_f1"],
        "Macro-F1",
        "Train vs. Val Macro-F1",
    ),
]

for fname, train_y, val_y, ylabel, title in plots:
    try:
        plt.figure()
        plt.plot(train_y, label="Train")
        plt.plot(val_y, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{title} (Dataset: SPR_BENCH)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"spr_bench_{fname}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {fname}: {e}")
        plt.close()

# Confusion matrix heat-map
try:
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix (Dataset: SPR_BENCH)")
    plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------- METRIC PRINT ------------------------------------- #
test_acc = (preds == gts).mean()
test_f1 = macro_f1(preds, gts, n_cls)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")
print(f"Final Test Macro-F1 : {test_f1:.4f}")
