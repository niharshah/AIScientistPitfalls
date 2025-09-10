import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    spr_data = experiment_data.get("learning_rate_search", {}).get("SPR_BENCH", {})
    lrs = list(spr_data.keys())[:5]  # safeguard: plot at most 5 curves

    epochs = (
        len(next(iter(spr_data.values()))["metrics"]["train_acc"]) if spr_data else 0
    )
    x = np.arange(1, epochs + 1)

    # helper to extract list safely
    def get_list(d, path):
        cur = d
        for p in path:
            cur = cur.get(p, [])
        return cur

    # ---------- accuracy ----------
    try:
        plt.figure()
        for lr in lrs:
            acc_tr = get_list(spr_data[lr], ["metrics", "train_acc"])
            acc_val = get_list(spr_data[lr], ["metrics", "val_acc"])
            plt.plot(x, acc_tr, label=f"{lr}-train")
            plt.plot(x, acc_val, "--", label=f"{lr}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy Curves\nLearning-Rate Search (Train vs. Val)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------- loss ----------
    try:
        plt.figure()
        for lr in lrs:
            loss_tr = get_list(spr_data[lr], ["losses", "train"])
            loss_val = get_list(spr_data[lr], ["losses", "val"])
            plt.plot(x, loss_tr, label=f"{lr}-train")
            plt.plot(x, loss_val, "--", label=f"{lr}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLearning-Rate Search (Train vs. Val)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- rule fidelity ----------
    try:
        plt.figure()
        for lr in lrs:
            rf = get_list(spr_data[lr], ["metrics", "rule_fidelity"])
            plt.plot(x, rf, label=lr)
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title("SPR_BENCH Rule Fidelity Across Epochs\nLearning-Rate Search")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating rule fidelity plot: {e}")
        plt.close()

    # ---------- print final test accuracy ----------
    for lr in lrs:
        test_acc = spr_data[lr]["metrics"].get("test_acc", None)
        if test_acc is not None:
            print(f"{lr}: Test Accuracy = {test_acc:.3f}")
