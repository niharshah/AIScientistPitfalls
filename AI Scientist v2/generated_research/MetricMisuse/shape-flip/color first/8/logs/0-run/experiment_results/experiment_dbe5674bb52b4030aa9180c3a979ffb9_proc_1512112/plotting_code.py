import matplotlib.pyplot as plt
import numpy as np
import os

# ----- set up work dir & load -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ----- collect summary stats -----
lrs, val_accs, compwas, best_lr, best_val = [], [], [], None, -1
for lr_str, log in experiment_data.get("learning_rate", {}).items():
    lrs.append(float(lr_str))
    val_acc = log["metrics"]["val"][-1] if log["metrics"]["val"] else 0.0
    compwa = log["metrics"].get("comp_weighted_accuracy", 0.0)
    val_accs.append(val_acc)
    compwas.append(compwa)
    if val_acc > best_val:
        best_val, best_lr = val_acc, float(lr_str)

# ----- print summary -----
print("Learning-Rate  |  Final Val Acc  |  Complexity-Weighted Acc")
for lr, v, c in zip(lrs, val_accs, compwas):
    print(f"{lr:10.4g} | {v:15.3f} | {c:25.3f}")

# ----------------- PLOTS -----------------
# 1) Validation accuracy by learning rate
try:
    plt.figure()
    plt.bar([str(lr) for lr in lrs], val_accs, color="skyblue")
    plt.title("SPR_BENCH: Final Validation Accuracy vs Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Validation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_accuracy_by_lr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val-acc plot: {e}")
    plt.close()

# 2) Complexity-weighted accuracy by learning rate
try:
    plt.figure()
    plt.bar([str(lr) for lr in lrs], compwas, color="lightgreen")
    plt.title("SPR_BENCH: Complexity-Weighted Accuracy vs Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_comp_weighted_accuracy_by_lr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CompWA plot: {e}")
    plt.close()

# 3) Loss curves for best learning rate
try:
    if best_lr is not None:
        log = experiment_data["learning_rate"][str(best_lr)]
        epochs = np.arange(1, len(log["losses"]["train"]) + 1)
        plt.figure()
        plt.plot(epochs, log["losses"]["train"], label="Train Loss")
        plt.plot(epochs, log["losses"]["val"], label="Val Loss")
        plt.title(f"SPR_BENCH: Loss Curves (Best LR={best_lr})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"SPR_BENCH_loss_curves_best_lr_{best_lr}.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating loss-curve plot: {e}")
    plt.close()
