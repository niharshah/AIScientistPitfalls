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
    ed = experiment_data["optimizer_type"]["SPR_BENCH"]
    cfgs = list(ed["configs"])
    train_acc = ed["metrics"]["train_acc"]
    val_acc = ed["metrics"]["val_acc"]
    rule_fid = ed["metrics"]["rule_fidelity"]
    train_loss = ed["losses"]["train"]
    val_loss = ed["losses"]["val"]
    preds = ed["predictions"]
    gts = ed["ground_truth"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

# ---------- ACCURACY PLOT ----------
try:
    plt.figure()
    for i, cfg in enumerate(cfgs):
        epochs = np.arange(1, len(train_acc[i]) + 1)
        plt.plot(epochs, train_acc[i], label=f"{cfg}-train")
        plt.plot(epochs, val_acc[i], "--", label=f"{cfg}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Training vs Validation Accuracy")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_train_val_accuracy.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- LOSS PLOT ----------
try:
    plt.figure()
    for i, cfg in enumerate(cfgs):
        epochs = np.arange(1, len(train_loss[i]) + 1)
        plt.plot(epochs, train_loss[i], label=f"{cfg}-train")
        plt.plot(epochs, val_loss[i], "--", label=f"{cfg}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Training vs Validation Loss")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_train_val_loss.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- RULE FIDELITY PLOT ----------
try:
    plt.figure()
    for i, cfg in enumerate(cfgs):
        epochs = np.arange(1, len(rule_fid[i]) + 1)
        plt.plot(epochs, rule_fid[i], label=f"{cfg}")
    plt.xlabel("Epoch")
    plt.ylabel("Rule Fidelity")
    plt.title("SPR_BENCH Rule Fidelity per Epoch")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating rule fidelity plot: {e}")
    plt.close()

# ---------- GROUND-TRUTH vs PREDICTION DISTRIBUTION ----------
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
    save_path = os.path.join(working_dir, "SPR_BENCH_gt_vs_pred_distribution.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating GT vs Pred plot: {e}")
    plt.close()

# ---------- PRINT TEST ACCURACY ----------
test_acc = (preds == gts).mean()
print(f"Best config: {ed['best_config']}  |  Test accuracy: {test_acc:.3f}")
