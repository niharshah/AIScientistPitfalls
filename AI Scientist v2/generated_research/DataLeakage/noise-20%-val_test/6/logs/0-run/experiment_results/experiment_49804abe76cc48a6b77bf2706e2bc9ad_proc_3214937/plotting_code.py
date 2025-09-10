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
    train_acc = np.asarray(ed["metrics"]["train"])
    val_acc = np.asarray(ed["metrics"]["val"])
    rule_fid = np.asarray(ed["metrics"]["rule_fid"])
    train_loss = np.asarray(ed["losses"]["train"])
    val_loss = np.asarray(ed["losses"]["val"])
    preds = np.asarray(ed["predictions"])
    gts = np.asarray(ed["ground_truth"])
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

epochs = np.arange(1, len(train_acc) + 1)

# ---------- ACCURACY PLOT ----------
try:
    plt.figure()
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, "--", label="Validation")
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
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, "--", label="Validation")
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
    plt.plot(epochs, rule_fid, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Rule Fidelity")
    plt.title("SPR_BENCH Rule Fidelity per Epoch")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png"))
    plt.close()
except Exception as e:
    print(f"Error creating rule fidelity plot: {e}")
    plt.close()

# ---------- GROUND-TRUTH vs PREDICTION DISTRIBUTION ----------
try:
    classes = np.sort(np.unique(np.concatenate([gts, preds])))
    gt_counts = np.array([(gts == c).sum() for c in classes])
    pred_counts = np.array([(preds == c).sum() for c in classes])

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
try:
    test_acc = (preds == gts).mean()
    print(f"SPR_BENCH Test Accuracy: {test_acc:.3f}")
except Exception as e:
    print(f"Error computing test accuracy: {e}")
