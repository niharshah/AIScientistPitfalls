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
    raise SystemExit

# ---------- PER DATASET VISUALISATIONS ----------
for dset, ed in experiment_data.items():
    train_acc_all = ed["metrics"]["train_acc"]  # list[run][epoch]
    val_acc_all = ed["metrics"]["val_acc"]
    rule_fid_all = ed["metrics"]["rule_fidelity"]  # list[run][epoch] OR list[run] ?
    train_loss_all = ed["losses"]["train"]
    val_loss_all = ed["losses"]["val"]
    preds = np.array(ed["predictions"])
    gts = np.array(ed["ground_truth"])
    num_runs = len(train_acc_all)

    # ---------- ACCURACY ----------
    try:
        plt.figure()
        for i in range(num_runs):
            ep = np.arange(1, len(train_acc_all[i]) + 1)
            plt.plot(ep, train_acc_all[i], label=f"run{i}-train")
            plt.plot(ep, val_acc_all[i], "--", label=f"run{i}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dset} Training vs Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_train_val_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dset}: {e}")
        plt.close()

    # ---------- LOSS ----------
    try:
        plt.figure()
        for i in range(num_runs):
            ep = np.arange(1, len(train_loss_all[i]) + 1)
            plt.plot(ep, train_loss_all[i], label=f"run{i}-train")
            plt.plot(ep, val_loss_all[i], "--", label=f"run{i}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset} Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_train_val_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
        plt.close()

    # ---------- RULE FIDELITY ----------
    try:
        if isinstance(rule_fid_all[0], list):  # stored per epoch
            plt.figure()
            for i in range(num_runs):
                ep = np.arange(1, len(rule_fid_all[i]) + 1)
                plt.plot(ep, rule_fid_all[i], label=f"run{i}")
            plt.xlabel("Epoch")
            plt.ylabel("Rule Fidelity")
            plt.title(f"{dset} Rule Fidelity per Epoch")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dset}_rule_fidelity.png"))
            plt.close()
        else:
            # single value per run
            plt.figure()
            plt.bar(np.arange(num_runs), rule_fid_all)
            plt.xlabel("Run")
            plt.ylabel("Rule Fidelity")
            plt.title(f"{dset} Final Rule Fidelity by Run")
            plt.savefig(os.path.join(working_dir, f"{dset}_rule_fidelity.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating rule fidelity plot for {dset}: {e}")
        plt.close()

    # ---------- GT vs PRED DISTRIBUTION ----------
    try:
        classes = np.sort(np.unique(np.concatenate([gts, preds])))
        gt_counts = np.array([np.sum(gts == c) for c in classes])
        pred_counts = np.array([np.sum(preds == c) for c in classes])
        x = np.arange(len(classes))
        bar_w = 0.4
        plt.figure(figsize=(8, 4))
        plt.bar(x - bar_w / 2, gt_counts, width=bar_w, label="Ground Truth")
        plt.bar(x + bar_w / 2, pred_counts, width=bar_w, label="Predicted")
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.title(
            f"{dset} Class Distribution\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dset}_gt_vs_pred_distribution.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating GT vs Pred plot for {dset}: {e}")
        plt.close()

    # ---------- TEST ACCURACY ----------
    try:
        if preds.size and gts.size:
            test_acc = (preds == gts).mean()
            print(f"{dset}: best-run test accuracy = {test_acc:.3f}")
    except Exception as e:
        print(f"Error computing test accuracy for {dset}: {e}")
