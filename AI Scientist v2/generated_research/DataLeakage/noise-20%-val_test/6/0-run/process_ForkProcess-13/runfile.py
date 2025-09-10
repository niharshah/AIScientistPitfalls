import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------ LOAD DATA ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("weight_decay", {}).get("SPR_BENCH", {})
if not runs:
    print("No data to plot.")
    exit()

decays = sorted(runs.keys(), key=lambda x: float(x))
epochs = len(next(iter(runs.values()))["metrics"]["train_acc"])

# collect metrics
val_accs, train_losses, val_losses, rule_fids, test_accs = {}, {}, {}, {}, {}
for d in decays:
    m = runs[d]["metrics"]
    val_accs[d] = m["val_acc"]
    train_losses[d] = runs[d]["losses"]["train"]
    val_losses[d] = runs[d]["losses"]["val"]
    rule_fids[d] = m["rule_fidelity"]
    test_accs[d] = runs[d]["test_acc"]

best_decay = max(test_accs, key=test_accs.get)
best_preds = runs[best_decay]["predictions"]
y_true = runs[best_decay]["ground_truth"]
num_classes = len(np.unique(y_true))

# -------------- PLOTTING -----------------------
# 1) Validation accuracy curves
try:
    plt.figure()
    for d in decays:
        plt.plot(range(1, epochs + 1), val_accs[d], label=f"wd={d}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("SPR_BENCH - Validation Accuracy vs Epoch (weight decays)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_accuracy_vs_epoch.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val-acc plot: {e}")
    plt.close()

# 2) Train & validation loss
try:
    plt.figure()
    for d in decays:
        plt.plot(
            range(1, epochs + 1), train_losses[d], linestyle="--", label=f"Train wd={d}"
        )
        plt.plot(range(1, epochs + 1), val_losses[d], label=f"Val wd={d}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH - Train & Val Loss vs Epoch (weight decays)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_vs_epoch.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) Rule fidelity
try:
    plt.figure()
    for d in decays:
        plt.plot(range(1, epochs + 1), rule_fids[d], label=f"wd={d}")
    plt.xlabel("Epoch")
    plt.ylabel("Rule Fidelity")
    plt.title("SPR_BENCH - Rule Fidelity vs Epoch (weight decays)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_rule_fidelity_vs_epoch.png"))
    plt.close()
except Exception as e:
    print(f"Error creating rule-fidelity plot: {e}")
    plt.close()

# 4) Final test accuracies
try:
    plt.figure()
    plt.bar(range(len(decays)), [test_accs[d] for d in decays], tick_label=decays)
    plt.ylabel("Test Accuracy")
    plt.title("SPR_BENCH - Final Test Accuracy per Weight Decay")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test-acc bar plot: {e}")
    plt.close()

# 5) Confusion matrix for best decay
try:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, best_preds):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"SPR_BENCH - Confusion Matrix (wd={best_decay})")
    plt.savefig(
        os.path.join(working_dir, f"SPR_BENCH_confusion_matrix_wd_{best_decay}.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# -------- Print numeric summary ----------
print("=== Final Test Accuracies ===")
for d in decays:
    print(f"weight_decay={d}: {test_accs[d]:.4f}")
