import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    ed = experiment_data["SPR_BENCH"]

    # --------- helpers -------------------------------------------------------
    def split_xy(tuples):
        xs, ys = zip(*tuples) if tuples else ([], [])
        return list(xs), list(ys)

    ep_loss_tr_x, ep_loss_tr_y = split_xy(ed["losses"]["train"])
    ep_loss_vl_x, ep_loss_vl_y = split_xy(ed["losses"]["val"])
    ep_swa_tr_x, ep_swa_tr_y = split_xy(ed["metrics"]["train"])
    ep_swa_vl_x, ep_swa_vl_y = split_xy(ed["metrics"]["val"])
    gt, pr = ed["ground_truth"], ed["predictions"]
    test_acc = np.mean(np.array(gt) == np.array(pr)) if gt else np.nan

    # --------- Figure 1 : Loss curves ---------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(ep_loss_tr_x, ep_loss_tr_y, label="Train")
        plt.plot(ep_loss_vl_x, ep_loss_vl_y, label="Validation")
        plt.title("SPR_BENCH – Cross-Entropy Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # --------- Figure 2 : SWA curves ----------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(ep_swa_tr_x, ep_swa_tr_y, label="Train")
        plt.plot(ep_swa_vl_x, ep_swa_vl_y, label="Validation")
        plt.title("SPR_BENCH – Shape-Weighted Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curves: {e}")
        plt.close()

    # --------- Figure 3 : Confusion matrix ----------------------------------
    try:
        from itertools import product

        cm = np.zeros((2, 2), dtype=int)
        for g, p in zip(gt, pr):
            cm[g][p] += 1
        plt.figure(figsize=(4, 3))
        plt.imshow(cm, cmap="Blues")
        plt.title("SPR_BENCH – Confusion Matrix (Test)")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        for i, j in product(range(2), range(2)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # --------- Final metric print ------------------------------------------
    print(f"SPR_BENCH – Test Accuracy: {test_acc:.4f}")
