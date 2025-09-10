import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["dropout_rate"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

# gather basic info
dropouts = list(exp.keys())
epochs = len(next(iter(exp.values()))["losses"]["train"]) if exp else 0
epoch_idx = np.arange(1, epochs + 1)

# helper: pick best dropout based on final val F1
best_dropout, best_f1 = None, -1.0
final_f1s = {}
for d in dropouts:
    val_f1 = exp[d]["metrics"]["val"][-1]
    final_f1s[d] = val_f1
    if val_f1 > best_f1:
        best_f1, best_dropout = val_f1, d

print("Final validation F1 per dropout:")
for d, f1 in final_f1s.items():
    print(f"  dropout={d}: {f1:.4f}")
print(f"Best dropout according to val F1: {best_dropout} ({best_f1:.4f})")

# 1) Loss curves
try:
    plt.figure(figsize=(7, 5))
    for d in dropouts:
        plt.plot(epoch_idx, exp[d]["losses"]["train"], label=f"train d={d}")
        plt.plot(epoch_idx, exp[d]["losses"]["val"], label=f"val d={d}", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) F1 curves
try:
    plt.figure(figsize=(7, 5))
    for d in dropouts:
        plt.plot(epoch_idx, exp[d]["metrics"]["train"], label=f"train d={d}")
        plt.plot(
            epoch_idx, exp[d]["metrics"]["val"], label=f"val d={d}", linestyle="--"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Training vs Validation F1")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_F1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# 3) Final F1 bar plot
try:
    plt.figure(figsize=(6, 4))
    plt.bar(
        range(len(dropouts)),
        [final_f1s[d] for d in dropouts],
        tick_label=dropouts,
        color="skyblue",
    )
    plt.ylabel("Final Validation Macro-F1")
    plt.title("SPR_BENCH: Final F1 vs Dropout Rate")
    fname = os.path.join(working_dir, "SPR_BENCH_final_F1_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final F1 bar: {e}")
    plt.close()

# 4) Confusion matrix for best model
try:
    gt = np.array(exp[best_dropout]["ground_truth"])
    prd = np.array(exp[best_dropout]["predictions"])
    num_classes = len(np.unique(gt))
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for g, p in zip(gt, prd):
        cm[g, p] += 1

    plt.figure(figsize=(5, 4))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"SPR_BENCH Confusion Matrix (dropout={best_dropout})")
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fname = os.path.join(
        working_dir, f"SPR_BENCH_confusion_matrix_dropout_{best_dropout}.png"
    )
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
