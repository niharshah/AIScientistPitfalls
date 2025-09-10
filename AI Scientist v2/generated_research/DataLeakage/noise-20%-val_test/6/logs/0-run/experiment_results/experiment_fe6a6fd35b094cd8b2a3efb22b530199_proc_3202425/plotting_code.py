import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup ---------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# quick sanity
if not experiment_data:
    print("No experiment data found, exiting.")
    exit()

ds_name = "SPR_BENCH"
runs = experiment_data["learning_rate"][ds_name]  # dict keyed by lr_*
lrs = list(runs.keys())

# gather metrics
epochs = len(next(iter(runs.values()))["metrics"]["train_acc"])
best_lr, best_val = None, -1
for lr_key, ed in runs.items():
    val_acc_last = ed["metrics"]["val_acc"][-1]
    if val_acc_last > best_val:
        best_val, best_lr = val_acc_last, lr_key


# ----------- plotting helpers -------------
def plot_curves(metric_key, ylabel, filename_suffix):
    try:
        plt.figure()
        for lr_key in lrs:
            epochs_x = np.arange(1, epochs + 1)
            plt.plot(epochs_x, runs[lr_key]["metrics"][metric_key], label=f"{lr_key}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ds_name}: {ylabel} vs Epoch")
        plt.legend()
        save_path = os.path.join(working_dir, f"{ds_name}_{filename_suffix}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating {filename_suffix} plot: {e}")
        plt.close()


# 1) Accuracy curves (train & val on same plot for clarity, limited to 4 lines *2)
try:
    plt.figure()
    epochs_x = np.arange(1, epochs + 1)
    for lr_key in lrs:
        plt.plot(
            epochs_x,
            runs[lr_key]["metrics"]["train_acc"],
            linestyle="--",
            label=f"{lr_key}_train",
        )
        plt.plot(epochs_x, runs[lr_key]["metrics"]["val_acc"], label=f"{lr_key}_val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{ds_name}: Training & Validation Accuracy")
    plt.legend(ncol=2, fontsize=7)
    plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_curves.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2) Loss curves
try:
    plt.figure()
    epochs_x = np.arange(1, epochs + 1)
    for lr_key in lrs:
        plt.plot(
            epochs_x,
            runs[lr_key]["losses"]["train"],
            linestyle="--",
            label=f"{lr_key}_train",
        )
        plt.plot(epochs_x, runs[lr_key]["losses"]["val"], label=f"{lr_key}_val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{ds_name}: Training & Validation Loss")
    plt.legend(ncol=2, fontsize=7)
    plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) Rule-fidelity curves
plot_curves("rule_fidelity", "Rule Fidelity", "rule_fidelity_curves")

# 4) Test accuracy bars
try:
    plt.figure()
    test_accs = [runs[lr_key]["test_acc"] for lr_key in lrs]
    plt.bar(range(len(lrs)), test_accs, tick_label=[k for k in lrs])
    plt.ylabel("Test Accuracy")
    plt.title(f"{ds_name}: Test Accuracy by Learning Rate")
    plt.savefig(os.path.join(working_dir, f"{ds_name}_test_accuracy.png"), dpi=300)
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar plot: {e}")
    plt.close()

# 5) Confusion matrix for best lr  (optional fifth plot)
try:
    import itertools

    ed_best = runs[best_lr]
    y_true = np.array(ed_best["ground_truth"])
    y_pred = np.array(ed_best["predictions"])
    num_classes = len(np.unique(y_true))
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"{ds_name}: Confusion Matrix (best {best_lr})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(num_classes), range(num_classes)):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=6,
        )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{ds_name}_confusion_matrix_{best_lr}.png"), dpi=300
    )
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

print("Finished plotting.")
