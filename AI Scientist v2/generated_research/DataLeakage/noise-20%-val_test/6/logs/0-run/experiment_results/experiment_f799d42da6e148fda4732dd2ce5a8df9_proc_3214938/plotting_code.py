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

# ---------- PER-DATASET PLOTTING ----------
for dname, ed in experiment_data.items():
    try:
        train_acc = ed["metrics"]["train_acc"]
        val_acc = ed["metrics"]["val_acc"]
        rule_fid = ed["metrics"]["rule_fidelity"]
        train_loss = ed["losses"]["train"]
        val_loss = ed["losses"]["val"]
        preds = np.array(ed["predictions"])
        gts = np.array(ed["ground_truth"])
        configs = [f"λ={lam}" for lam in ed.get("lambdas", [None] * len(train_acc))]
    except Exception as e:
        print(f"Missing keys for {dname}: {e}")
        continue

    # ---------- ACCURACY ----------
    try:
        plt.figure()
        for i, cfg in enumerate(configs):
            epochs = np.arange(1, len(train_acc[i]) + 1)
            plt.plot(epochs, train_acc[i], label=f"{cfg}-train")
            plt.plot(epochs, val_acc[i], "--", label=f"{cfg}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dname} Training vs Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_train_val_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dname}: {e}")
        plt.close()

    # ---------- LOSS ----------
    try:
        plt.figure()
        for i, cfg in enumerate(configs):
            epochs = np.arange(1, len(train_loss[i]) + 1)
            plt.plot(epochs, train_loss[i], label=f"{cfg}-train")
            plt.plot(epochs, val_loss[i], "--", label=f"{cfg}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dname} Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_train_val_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # ---------- RULE FIDELITY ----------
    try:
        plt.figure()
        for i, cfg in enumerate(configs):
            epochs = np.arange(1, len(rule_fid[i]) + 1)
            plt.plot(epochs, rule_fid[i], label=cfg)
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title(f"{dname} Rule Fidelity per Epoch")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_rule_fidelity.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating rule fidelity plot for {dname}: {e}")
        plt.close()

    # ---------- CLASS DISTRIBUTION ----------
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
            f"{dname} Class Distribution\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_gt_vs_pred_distribution.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating GT vs Pred plot for {dname}: {e}")
        plt.close()

    # ---------- PRINT TEST ACCURACY ----------
    try:
        test_acc = (preds == gts).mean()
        print(f"{dname}: Test accuracy={test_acc:.3f} | Best λ={ed.get('best_lambda')}")
    except Exception as e:
        print(f"Error computing test accuracy for {dname}: {e}")
