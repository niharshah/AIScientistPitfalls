import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- LOAD DATA ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["weight_decay"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

train_acc = ed["metrics"]["train_acc"]
val_acc = ed["metrics"]["val_acc"]
train_loss = ed["losses"]["train"]
val_loss = ed["metrics"]["val_loss"]
wds = ed["wds"]
test_acc = ed["test_acc"]
fidelity = ed["fidelity"]
fagm = ed["fagm"]
EPOCHS = len(train_acc) // len(wds) if wds else 0


# Helper to slice per-run lists
def slice_per_run(lst):
    return [lst[i * EPOCHS : (i + 1) * EPOCHS] for i in range(len(wds))]


train_acc_runs = slice_per_run(train_acc)
val_acc_runs = slice_per_run(val_acc)
train_loss_runs = slice_per_run(train_loss)
val_loss_runs = slice_per_run(val_loss)

# ---------------- PLOTTING ----------------
# 1) Accuracy curves
try:
    plt.figure()
    for i, wd in enumerate(wds):
        epochs = np.arange(1, EPOCHS + 1)
        plt.plot(epochs, train_acc_runs[i], label=f"train_acc wd={wd}")
        plt.plot(epochs, val_acc_runs[i], "--", label=f"val_acc wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy vs Epoch for different weight_decays")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2) Loss curves
try:
    plt.figure()
    for i, wd in enumerate(wds):
        epochs = np.arange(1, EPOCHS + 1)
        plt.plot(epochs, train_loss_runs[i], label=f"train_loss wd={wd}")
        plt.plot(epochs, val_loss_runs[i], "--", label=f"val_loss wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss vs Epoch for different weight_decays")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) Test metrics bar chart
try:
    x = np.arange(len(wds))
    width = 0.25
    plt.figure()
    plt.bar(x - width, test_acc, width, label="test_acc")
    plt.bar(x, fidelity, width, label="fidelity")
    plt.bar(x + width, fagm, width, label="FAGM")
    plt.xticks(x, [str(wd) for wd in wds])
    plt.xlabel("weight_decay")
    plt.ylabel("Score")
    plt.title("SPR_BENCH Final Metrics vs weight_decay")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric bar chart: {e}")
    plt.close()

# 4) Confusion matrix for best FAGM
try:
    best_idx = int(np.argmax(fagm))
    preds = ed["predictions"][best_idx]
    gts = ed["ground_truth"][best_idx]
    n_cls = max(gts.max(), preds.max()) + 1
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for p, t in zip(preds, gts):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"SPR_BENCH Confusion Matrix (wd={wds[best_idx]})")
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix_best_wd.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------------- PRINT SUMMARY ----------------
header = ["wd", "test_acc", "fidelity", "FAGM"]
print("\t".join(header))
for i, wd in enumerate(wds):
    print(f"{wd}\t{test_acc[i]:.3f}\t{fidelity[i]:.3f}\t{fagm[i]:.3f}")
print(f"Best weight_decay by FAGM: {wds[best_idx]} (FAGM={fagm[best_idx]:.3f})")
