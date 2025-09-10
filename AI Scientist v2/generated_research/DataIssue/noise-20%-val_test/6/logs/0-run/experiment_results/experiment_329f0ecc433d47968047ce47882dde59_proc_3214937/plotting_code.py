import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- SETUP ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- LOAD EXP DATA ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

lambdas = ed["lambdas"]
cfgs = [f"λ={lam}" for lam in lambdas]
train_acc = ed["metrics"]["train_acc"]
val_acc = ed["metrics"]["val_acc"]
rule_fid = ed["metrics"]["RFA"]
train_loss = ed["losses"]["train"]
val_loss = ed["losses"]["val"]
preds = ed["predictions"]
gts = ed["ground_truth"]

# ---------- ACCURACY PLOT ----------
try:
    plt.figure()
    for i, label in enumerate(cfgs):
        epochs = np.arange(1, len(train_acc[i]) + 1)
        plt.plot(epochs, train_acc[i], label=f"{label}-train")
        plt.plot(epochs, val_acc[i], "--", label=f"{label}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- LOSS PLOT ----------
try:
    plt.figure()
    for i, label in enumerate(cfgs):
        epochs = np.arange(1, len(train_loss[i]) + 1)
        plt.plot(epochs, train_loss[i], label=f"{label}-train")
        plt.plot(epochs, val_loss[i], "--", label=f"{label}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- RULE FIDELITY PLOT ----------
try:
    plt.figure()
    for i, label in enumerate(cfgs):
        epochs = np.arange(1, len(rule_fid[i]) + 1)
        plt.plot(epochs, rule_fid[i], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Rule Fidelity (Top-10)")
    plt.title("SPR_BENCH Rule Fidelity per Epoch")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png"))
    plt.close()
except Exception as e:
    print(f"Error creating rule fidelity plot: {e}")
    plt.close()

# ---------- GROUND-TRUTH vs PRED DISTRIBUTION ----------
try:
    classes = np.sort(np.unique(np.concatenate([gts, preds])))
    gt_counts = np.array([np.sum(gts == c) for c in classes])
    pred_counts = np.array([np.sum(preds == c) for c in classes])

    bar_w = 0.4
    x = np.arange(len(classes))
    plt.figure(figsize=(8, 4))
    plt.bar(x - bar_w / 2, gt_counts, width=bar_w, label="Ground Truth")
    plt.bar(x + bar_w / 2, pred_counts, width=bar_w, label="Predicted")
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.title(
        "SPR_BENCH Class Distribution\nLeft: Ground Truth, Right: Generated Samples"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_gt_vs_pred_distribution.png"))
    plt.close()
except Exception as e:
    print(f"Error creating GT vs Pred plot: {e}")
    plt.close()

# ---------- PRINT TEST ACCURACY ----------
test_acc = (preds == gts).mean()
print(f"Best λ: {ed['best_lambda']}  |  Test accuracy: {test_acc:.3f}")
