import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- LOAD EXPERIMENT DATA ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to filter learning-rate entries
def get_lr_tags(edict):
    return [
        k
        for k in edict.keys()
        if k
        not in {
            "best_lr",
            "test_acc",
            "predictions",
            "ground_truth",
            "rule_preds",
            "rule_fidelity",
            "fagm",
        }
    ]


ds_name = "SPR_BENCH"
lr_branch = experiment_data.get("learning_rate", {}).get(ds_name, {})
lr_tags = get_lr_tags(lr_branch)

# ---------- PLOT 1: ACCURACY CURVES ----------
try:
    plt.figure()
    for tag in lr_tags:
        epochs = range(1, len(lr_branch[tag]["metrics"]["train_acc"]) + 1)
        plt.plot(
            epochs, lr_branch[tag]["metrics"]["train_acc"], label=f"train_acc lr={tag}"
        )
        plt.plot(
            epochs,
            lr_branch[tag]["metrics"]["val_acc"],
            linestyle="--",
            label=f"val_acc lr={tag}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{ds_name}: Training vs Validation Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name}_train_val_acc_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- PLOT 2: LOSS CURVES ----------
try:
    plt.figure()
    for tag in lr_tags:
        epochs = range(1, len(lr_branch[tag]["losses"]["train"]) + 1)
        plt.plot(
            epochs, lr_branch[tag]["losses"]["train"], label=f"train_loss lr={tag}"
        )
        plt.plot(
            epochs,
            lr_branch[tag]["metrics"]["val_loss"],
            linestyle="--",
            label=f"val_loss lr={tag}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{ds_name}: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name}_train_val_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- PLOT 3: FINAL VAL ACC BY LR ----------
try:
    plt.figure()
    vals = [lr_branch[tag]["metrics"]["val_acc"][-1] for tag in lr_tags]
    plt.bar(lr_tags, vals, color="skyblue")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Validation Accuracy")
    plt.title(f"{ds_name}: Final Validation Accuracy per Learning Rate")
    fname = os.path.join(working_dir, f"{ds_name}_val_acc_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val-acc bar plot: {e}")
    plt.close()

# ---------- PLOT 4: TEST METRICS ----------
try:
    plt.figure()
    labels = ["Test Accuracy", "Rule Fidelity", "FAGM"]
    metrics = [
        lr_branch.get("test_acc", np.nan),
        lr_branch.get("rule_fidelity", np.nan),
        lr_branch.get("fagm", np.nan),
    ]
    plt.bar(labels, metrics, color=["seagreen", "orange", "purple"])
    plt.ylim(0, 1)
    plt.title(f"{ds_name}: Best Model – Test Metrics")
    fname = os.path.join(working_dir, f"{ds_name}_best_model_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating best-model metrics plot: {e}")
    plt.close()

# ---------- PRINT KEY METRICS ----------
best_lr = lr_branch.get("best_lr", None)
test_acc = lr_branch.get("test_acc", None)
fidelity = lr_branch.get("rule_fidelity", None)
fagm = lr_branch.get("fagm", None)
print(
    f"Best LR: {best_lr} | Test Acc: {test_acc} | Rule Fidelity: {fidelity} | FAGM: {fagm}"
)
