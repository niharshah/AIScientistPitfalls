import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    spr = experiment_data["SPR_BENCH"]

    # ----------------------------- extract curves ----------------
    tr_loss_tuples = spr["losses"]["train"]
    vl_loss_tuples = spr["losses"]["val"]
    tr_swa_tuples = spr["metrics"]["train"]
    vl_swa_tuples = spr["metrics"]["val"]

    epochs = [e for e, _ in tr_loss_tuples]
    loss_tr = [v for _, v in tr_loss_tuples]
    loss_vl = [v for _, v in vl_loss_tuples]
    swa_tr = [v for _, v in tr_swa_tuples]
    swa_vl = [v for _, v in vl_swa_tuples]

    # ----------------------------- figure 1 ----------------------
    try:
        plt.figure()
        plt.plot(epochs, loss_tr, label="Train")
        plt.plot(epochs, loss_vl, label="Validation")
        plt.title("SPR_BENCH – Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ----------------------------- figure 2 ----------------------
    try:
        plt.figure()
        plt.plot(epochs, swa_tr, label="Train")
        plt.plot(epochs, swa_vl, label="Validation")
        plt.title("SPR_BENCH – Shape-Weighted Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_swa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # ----------------------------- test accuracy -----------------
    preds = spr.get("predictions", [])
    gts = spr.get("ground_truth", [])
    if preds and gts and len(preds) == len(gts):
        test_acc = sum(int(p == g) for p, g in zip(preds, gts)) / len(gts)
    else:
        test_acc = np.nan

    # ----------------------------- figure 3 ----------------------
    try:
        plt.figure()
        plt.bar(["SPR_BENCH"], [test_acc])
        plt.title("SPR_BENCH – Test Accuracy")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.text(0, test_acc + 0.02, f"{test_acc:.2f}", ha="center")
        fname = os.path.join(working_dir, "spr_bench_test_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy plot: {e}")
        plt.close()

    # ----------------------------- print metric ------------------
    print(f"SPR_BENCH Test Accuracy: {test_acc:.4f}")
else:
    print("SPR_BENCH data not found in experiment_data.")
