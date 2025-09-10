import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment data ---------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    wd_keys = sorted(experiment_data["weight_decay"].keys(), key=float)
    # Gather metrics
    dev_curves = {
        wd: experiment_data["weight_decay"][wd]["metrics"]["dev"] for wd in wd_keys
    }
    test_acc = {
        wd: experiment_data["weight_decay"][wd]["metrics"]["test"]["acc"]
        for wd in wd_keys
    }
    swa = {
        wd: experiment_data["weight_decay"][wd]["metrics"]["test"]["swa"]
        for wd in wd_keys
    }
    cwa = {
        wd: experiment_data["weight_decay"][wd]["metrics"]["test"]["cwa"]
        for wd in wd_keys
    }

    # 1) Dev accuracy curves
    try:
        plt.figure()
        for wd, vals in dev_curves.items():
            plt.plot(range(1, len(vals) + 1), vals, label=f"wd={wd}")
        plt.title("Synthetic SPR – Dev Accuracy vs Epochs (all weight decays)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "synthetic_spr_dev_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating dev accuracy plot: {e}")
        plt.close()

    # 2) Test accuracy bar chart
    try:
        plt.figure()
        plt.bar(
            range(len(wd_keys)), [test_acc[wd] for wd in wd_keys], tick_label=wd_keys
        )
        plt.title("Synthetic SPR – Test Accuracy by Weight Decay")
        plt.xlabel("Weight Decay")
        plt.ylabel("Accuracy")
        fname = os.path.join(working_dir, "synthetic_spr_test_accuracy_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy bar: {e}")
        plt.close()

    # 3) Shape-Weighted Accuracy
    try:
        plt.figure()
        plt.bar(range(len(wd_keys)), [swa[wd] for wd in wd_keys], tick_label=wd_keys)
        plt.title("Synthetic SPR – Shape-Weighted Accuracy (SWA) by Weight Decay")
        plt.xlabel("Weight Decay")
        plt.ylabel("SWA")
        fname = os.path.join(working_dir, "synthetic_spr_swa_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA bar: {e}")
        plt.close()

    # 4) Color-Weighted Accuracy
    try:
        plt.figure()
        plt.bar(range(len(wd_keys)), [cwa[wd] for wd in wd_keys], tick_label=wd_keys)
        plt.title("Synthetic SPR – Color-Weighted Accuracy (CWA) by Weight Decay")
        plt.xlabel("Weight Decay")
        plt.ylabel("CWA")
        fname = os.path.join(working_dir, "synthetic_spr_cwa_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CWA bar: {e}")
        plt.close()

    # ------------------- print best configuration ---------------------
    best_wd = max(test_acc, key=test_acc.get)
    print(f"Best weight_decay={best_wd} with test accuracy={test_acc[best_wd]:.3f}")
