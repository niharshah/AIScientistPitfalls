import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------- #
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# -------------------------------------------------------------------------- #
def get_spr_info(edict):
    nh_vals = [cfg["nhead"] for cfg in edict["configs"]]
    t_acc, v_acc, s_acc = (
        edict["metrics"][k] for k in ("train_acc", "val_acc", "test_acc")
    )
    t_loss, v_loss, s_loss = (
        edict["losses"][k] for k in ("train_loss", "val_loss", "test_loss")
    )
    return nh_vals, t_acc, v_acc, s_acc, t_loss, v_loss, s_loss


if experiment_data:
    ed = experiment_data["nhead_tuning"]["SPR_BENCH"]
    nheads, tr_acc, val_acc, te_acc, tr_loss, val_loss, te_loss = get_spr_info(ed)

    # print quick table
    print("nhead | train_acc | val_acc | test_acc")
    for nh, ta, va, tsa in zip(nheads, tr_acc, val_acc, te_acc):
        print(f"{nh:5d} | {ta:.3f}     | {va:.3f}   | {tsa:.3f}")
    best_idx = int(np.argmax(val_acc))
    print(
        f"\nBest nhead={nheads[best_idx]} | val_acc={val_acc[best_idx]:.3f} | test_acc={te_acc[best_idx]:.3f}"
    )

    # ------------------ Plot 1: accuracy vs nhead -------------------------- #
    try:
        plt.figure()
        plt.plot(nheads, tr_acc, "o-", label="Train")
        plt.plot(nheads, val_acc, "s-", label="Validation")
        plt.plot(nheads, te_acc, "d-", label="Test")
        plt.xlabel("nhead")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Accuracy vs nhead")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_vs_nhead.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ------------------ Plot 2: loss vs nhead ------------------------------ #
    try:
        plt.figure()
        plt.plot(nheads, tr_loss, "o-", label="Train")
        plt.plot(nheads, val_loss, "s-", label="Validation")
        plt.plot(nheads, te_loss, "d-", label="Test")
        plt.xlabel("nhead")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Loss vs nhead")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_vs_nhead.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------------ Plot 3: val vs test scatter ------------------------ #
    try:
        plt.figure()
        plt.scatter(val_acc, te_acc, c="blue")
        plt.scatter(val_acc[best_idx], te_acc[best_idx], c="red", label="Best val")
        for i, nh in enumerate(nheads):
            plt.annotate(str(nh), (val_acc[i], te_acc[i]))
        plt.xlabel("Validation Accuracy")
        plt.ylabel("Test Accuracy")
        plt.title("SPR_BENCH: Validation vs Test Accuracy per nhead")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_vs_test_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        plt.close()
