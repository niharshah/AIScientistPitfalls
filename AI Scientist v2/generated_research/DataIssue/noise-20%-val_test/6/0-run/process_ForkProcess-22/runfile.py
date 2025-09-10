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
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- LOOP OVER DATASETS ----------
for ds_name, ed in experiment_data.items():
    # unpack data assuming simple no-config structure
    train_acc = np.asarray(ed["metrics"]["train_acc"])
    val_acc = np.asarray(ed["metrics"]["val_acc"])
    rule_fid = np.asarray(ed["metrics"]["rule_fid"])
    train_loss = np.asarray(ed["losses"]["train"])
    val_loss = np.asarray(ed["losses"]["val"])
    preds = np.asarray(ed["predictions"])
    gts = np.asarray(ed["ground_truth"])
    epochs = np.arange(1, len(train_acc) + 1)

    # ---------- ACCURACY PLOT ----------
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="train")
        plt.plot(epochs, val_acc, "--", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} Training vs Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_train_val_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds_name}: {e}")
        plt.close()

    # ---------- LOSS PLOT ----------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="train")
        plt.plot(epochs, val_loss, "--", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_train_val_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # ---------- RULE FIDELITY PLOT ----------
    try:
        plt.figure()
        plt.plot(epochs, rule_fid, label="rule fidelity")
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title(f"{ds_name} Rule Fidelity per Epoch")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_rule_fidelity.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating rule fidelity plot for {ds_name}: {e}")
        plt.close()

    # ---------- GT vs PRED DISTRIBUTION ----------
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
            f"{ds_name} Class Distribution\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_gt_vs_pred_distribution.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating GT vs Pred plot for {ds_name}: {e}")
        plt.close()

    # ---------- PRINT TEST ACCURACY ----------
    test_acc = (preds == gts).mean()
    print(f"{ds_name} test accuracy: {test_acc:.3f}")
