import matplotlib.pyplot as plt
import numpy as np
import os

# setup and data load
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# guard if data missing
per_lr = experiment_data.get("learning_rate", {}).get("SPR_BENCH", {}).get("per_lr", {})
if not per_lr:
    print("No experiment logs found – nothing to plot.")
    exit()

lrs = sorted(per_lr.keys(), key=lambda x: float(x))
epochs = range(1, len(next(iter(per_lr.values()))["losses"]["train"]) + 1)


# helper to gather metric over epochs
def gather(metric_key, phase):
    return {
        lr: [
            m[metric_key] if isinstance(m, dict) else m
            for m in per_lr[lr]["metrics" if metric_key != "loss" else "losses"][phase]
        ]
        for lr in lrs
    }


loss_tr = {lr: per_lr[lr]["losses"]["train"] for lr in lrs}
loss_val = {lr: per_lr[lr]["losses"]["val"] for lr in lrs}
acc_tr = gather("acc", "train")
acc_val = gather("acc", "val")
cowa_tr = gather("cowa", "train")
cowa_val = gather("cowa", "val")

# 1. Loss curves
try:
    plt.figure()
    for lr in lrs:
        plt.plot(epochs, loss_tr[lr], "--", label=f"train lr={lr}")
        plt.plot(epochs, loss_val[lr], "-", label=f"val lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: dashed=Train, Right: solid=Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2. Accuracy curves
try:
    plt.figure()
    for lr in lrs:
        plt.plot(epochs, acc_tr[lr], "--", label=f"train lr={lr}")
        plt.plot(epochs, acc_val[lr], "-", label=f"val lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy Curves\nLeft: dashed=Train, Right: solid=Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# 3. CoWA curves
try:
    plt.figure()
    for lr in lrs:
        plt.plot(epochs, cowa_tr[lr], "--", label=f"train lr={lr}")
        plt.plot(epochs, cowa_val[lr], "-", label=f"val lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("CoWA")
    plt.title(
        "SPR_BENCH Complexity-Weighted Accuracy (CoWA)\nLeft: dashed=Train, Right: solid=Validation"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_cowa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CoWA curves: {e}")
    plt.close()

# 4. Bar chart of final validation accuracy
try:
    final_val_acc = [acc_val[lr][-1] for lr in lrs]
    plt.figure()
    plt.bar(lrs, final_val_acc, color="skyblue")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Validation Accuracy")
    plt.title("SPR_BENCH Final Validation Accuracy per Learning Rate")
    fname = os.path.join(working_dir, "SPR_BENCH_final_val_acc_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# print best LR
best_idx = int(np.argmax(final_val_acc))
print(
    f"Best LR based on final val acc: {lrs[best_idx]} "
    f"(val_acc={final_val_acc[best_idx]:.3f})"
)
