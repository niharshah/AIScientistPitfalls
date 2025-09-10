import matplotlib.pyplot as plt
import numpy as np
import os

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

runs = experiment_data.get("weight_decay_tuning", {})
if not runs:
    print("No runs found in experiment_data.npy")
    exit()

wds = sorted(runs.keys(), key=float)
epochs = np.arange(1, len(next(iter(runs.values()))["losses"]["train"]) + 1)


# ---------- helper to fetch arrays ----------
def col(metric_path):
    # metric_path like ('losses','train') or ('metrics','val_acc')
    d1, d2 = metric_path
    return [runs[wd][d1][d2] for wd in wds]


train_loss = col(("losses", "train"))
train_acc = col(("metrics", "train_acc"))
val_acc = col(("metrics", "val_acc"))
val_ura = col(("metrics", "val_ura"))
test_acc = [runs[wd]["metrics"]["test_acc"] for wd in wds]
test_ura = [runs[wd]["metrics"]["test_ura"] for wd in wds]

# ---------- plotting ----------
plots_info = [
    ("SPR_BENCH_train_loss.png", train_loss, "Training Loss"),
    ("SPR_BENCH_train_val_acc.png", None, "Accuracy"),
    ("SPR_BENCH_val_ura.png", val_ura, "Validation URA"),
]

# 1) Train Loss
try:
    plt.figure()
    for wd, y in zip(wds, train_loss):
        plt.plot(epochs, y, label=f"wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, plots_info[0][0]))
    plt.close()
except Exception as e:
    print(f"Error creating train-loss plot: {e}")
    plt.close()

# 2) Train & Val Accuracy
try:
    plt.figure()
    for wd, y_tr, y_val in zip(wds, train_acc, val_acc):
        plt.plot(epochs, y_tr, label=f"train wd={wd}")
        plt.plot(epochs, y_val, "--", label=f"val wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training (solid) & Validation (dashed) Accuracy")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, plots_info[1][0]))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 3) Validation URA
try:
    plt.figure()
    for wd, y in zip(wds, val_ura):
        plt.plot(epochs, y, label=f"wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("URA")
    plt.title("Validation Unseen-Rule Accuracy (URA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, plots_info[2][0]))
    plt.close()
except Exception as e:
    print(f"Error creating URA plot: {e}")
    plt.close()

# 4) Test Accuracy bar chart
try:
    plt.figure()
    plt.bar(range(len(wds)), test_acc, tick_label=wds)
    plt.ylabel("Test Accuracy")
    plt.title("Final Test Accuracy per Weight Decay")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test-accuracy bar: {e}")
    plt.close()

# 5) Test URA bar chart
try:
    plt.figure()
    plt.bar(range(len(wds)), test_ura, tick_label=wds)
    plt.ylabel("Test URA")
    plt.title("Final Test URA per Weight Decay")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_URA.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test-URA bar: {e}")
    plt.close()

# ---------- best runs summary ----------
best_val_idx = int(np.argmax([v[-1] for v in val_acc]))
best_test_idx = int(np.argmax(test_acc))
print(
    f"Best val-acc  : wd={wds[best_val_idx]}, val_acc={val_acc[best_val_idx][-1]:.3f}"
)
print(
    f"Best test-acc : wd={wds[best_test_idx]}, test_acc={test_acc[best_test_idx]:.3f}"
)
