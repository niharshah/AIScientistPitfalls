import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load data -------------
try:
    edict = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    edict = edict["max_grad_norm"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    edict = None

if edict:

    hp_list = edict["hyperparams"]
    train_losses = edict["losses"]["train"]  # list[list]
    val_losses = edict["losses"]["val"]
    val_rcwas = edict["metrics"]["val"]
    preds_list = edict["predictions"]
    gt_list = edict["ground_truth"]

    # --------- 1) loss curves ----------
    try:
        plt.figure(figsize=(6, 4))
        for hp, tl, vl in zip(hp_list, train_losses, val_losses):
            epochs = np.arange(1, len(tl) + 1)
            plt.plot(epochs, tl, label=f"train mg={hp}")
            plt.plot(epochs, vl, "--", label=f"val mg={hp}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title("SPR_BENCH Loss Curves (Train vs Validation)")
        plt.legend(fontsize=8)
        save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # --------- 2) RCWA curves ----------
    try:
        plt.figure(figsize=(6, 4))
        for hp, vr in zip(hp_list, val_rcwas):
            if all(np.isnan(vr)):
                continue
            epochs = np.arange(1, len(vr) + 1)
            plt.plot(epochs, vr, label=f"val RCWA mg={hp}")
        plt.xlabel("Epoch")
        plt.ylabel("RCWA")
        plt.title("SPR_BENCH Validation RCWA across Epochs")
        plt.legend(fontsize=8)
        save_path = os.path.join(working_dir, "SPR_BENCH_val_RCWA_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating RCWA curves: {e}")
        plt.close()

    # --------- 3) final test RCWA bar ----------
    try:
        final_test_rcwa = []
        for preds, gts in zip(preds_list, gt_list):
            if len(preds) == 0:
                final_test_rcwa.append(np.nan)
                continue
            weights = [1] * len(
                preds
            )  # rcwa already computed earlier; but here create simple accuracy weight to reuse rcwa fn
            final_test_rcwa.append(
                (preds == gts).mean()
            )  # simple accuracy as placeholder
        x = np.arange(len(hp_list))
        plt.figure(figsize=(5, 3))
        plt.bar(x, final_test_rcwa, tick_label=[str(h) for h in hp_list])
        plt.ylabel("Test Accuracy (proxy)")
        plt.xlabel("max_grad_norm")
        plt.title("SPR_BENCH Test Accuracy vs Gradient Clipping")
        save_path = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating test bar plot: {e}")
        plt.close()

print(f"Plots saved to {working_dir}")
