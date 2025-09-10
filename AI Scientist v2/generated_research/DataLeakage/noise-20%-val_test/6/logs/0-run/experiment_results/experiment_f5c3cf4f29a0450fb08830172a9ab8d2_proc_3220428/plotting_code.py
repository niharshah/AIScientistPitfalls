import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load experiment results ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("no_bias_log_reg", {}).get("SPR_BENCH", None)

if ed:
    cfgs = ed["configs"]
    metrics = ed["metrics"]
    losses = ed["losses"]
    epochs = np.arange(1, len(metrics["train_acc"][0]) + 1)

    # --------- accuracy curves ----------
    try:
        plt.figure()
        for c, tr, va in zip(cfgs, metrics["train_acc"], metrics["val_acc"]):
            plt.plot(epochs, tr, label=f"{c}-train")
            plt.plot(epochs, va, "--", label=f"{c}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Training vs Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # --------- loss curves ----------
    try:
        plt.figure()
        for c, tr, va in zip(cfgs, losses["train"], losses["val"]):
            plt.plot(epochs, tr, label=f"{c}-train")
            plt.plot(epochs, va, "--", label=f"{c}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # --------- rule fidelity ----------
    try:
        plt.figure()
        for c, rf in zip(cfgs, metrics["rule_fidelity"]):
            plt.plot(epochs, rf, label=c)
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title("SPR_BENCH: Rule Fidelity over Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_rule_fidelity_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating rule fidelity plot: {e}")
        plt.close()

    # --------- confusion matrix ----------
    try:
        preds = np.asarray(ed["predictions"])
        gts = np.asarray(ed["ground_truth"])
        if preds.size == gts.size:
            n_cls = len(np.unique(gts))
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            cm_pct = cm / cm.sum(axis=1, keepdims=True)
            plt.figure()
            im = plt.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title("SPR_BENCH: Confusion Matrix (Best Config)")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            for i in range(n_cls):
                for j in range(n_cls):
                    plt.text(
                        j,
                        i,
                        f"{cm_pct[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="red" if cm_pct[i, j] > 0.5 else "black",
                        fontsize=8,
                    )
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # --------- print key metrics ----------
    try:
        best_idx = cfgs.index(ed["best_config"])
        dev_acc = metrics["val_acc"][best_idx][-1]
        test_acc = (preds == gts).mean()
        print(f"Best config: {ed['best_config']}")
        print(f"Dev accuracy (last epoch): {dev_acc:.3f}")
        print(f"Test accuracy: {test_acc:.3f}")
    except Exception as e:
        print(f"Error printing metrics: {e}")
else:
    print("No experiment data found for SPR_BENCH.")
