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
    exit()

# ---------- PLOTTING ----------
for dset, ed in experiment_data.items():
    # unpack
    train_acc = np.asarray(ed["metrics"]["train_acc"])
    val_acc = np.asarray(ed["metrics"]["val_acc"])
    rfa = np.asarray(ed["metrics"]["rfa"])
    train_ls = np.asarray(ed["losses"]["train"])
    val_ls = np.asarray(ed["losses"]["val"])
    preds = np.asarray(ed["predictions"])
    gts = np.asarray(ed["ground_truth"])

    epochs = np.arange(1, len(train_acc) + 1)
    # limit to at most 50 epochs in plots
    if len(epochs) > 50:
        step = len(epochs) // 50 + 1
        epochs = epochs[::step]

    # 1) ACCURACY PLOT
    try:
        plt.figure()
        plt.plot(epochs, train_acc[: len(epochs)], label="Train")
        plt.plot(epochs, val_acc[: len(epochs)], "--", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dset} Training vs Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_train_val_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dset}: {e}")
        plt.close()

    # 2) LOSS PLOT
    try:
        plt.figure()
        plt.plot(epochs, train_ls[: len(epochs)], label="Train")
        plt.plot(epochs, val_ls[: len(epochs)], "--", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset} Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_train_val_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
        plt.close()

    # 3) RULE FIDELITY PLOT
    try:
        plt.figure()
        plt.plot(epochs, rfa[: len(epochs)])
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title(f"{dset} Rule Fidelity per Epoch")
        plt.savefig(os.path.join(working_dir, f"{dset}_rule_fidelity.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating rule fidelity plot for {dset}: {e}")
        plt.close()

    # 4) GT vs PRED DISTRIBUTION
    try:
        classes = np.sort(np.unique(np.concatenate([gts, preds])))
        gt_cnt = np.array([np.sum(gts == c) for c in classes])
        pr_cnt = np.array([np.sum(preds == c) for c in classes])
        bar_w = 0.4
        x = np.arange(len(classes))
        plt.figure(figsize=(8, 4))
        plt.bar(x - bar_w / 2, gt_cnt, width=bar_w, label="Ground Truth")
        plt.bar(x + bar_w / 2, pr_cnt, width=bar_w, label="Predicted")
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.title(
            f"{dset} Class Distribution\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_gt_vs_pred_distribution.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating distribution plot for {dset}: {e}")
        plt.close()

    # ---------- PRINT TEST ACCURACY ----------
    try:
        test_acc = (preds == gts).mean()
        print(f"{dset}: Test Accuracy = {test_acc:.3f}")
    except Exception as e:
        print(f"Error computing test accuracy for {dset}: {e}")
