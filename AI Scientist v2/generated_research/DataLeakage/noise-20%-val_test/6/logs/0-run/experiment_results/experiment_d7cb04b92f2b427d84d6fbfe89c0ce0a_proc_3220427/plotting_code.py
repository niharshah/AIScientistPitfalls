import matplotlib.pyplot as plt
import numpy as np
import os

# IO --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Short-circuit if loading failed
if experiment_data:
    ds_key = "len_norm_ngram"
    dataset = "SPR_BENCH"
    ed = experiment_data[ds_key][dataset]
    cfgs = ed["configs"]
    epochs = len(ed["metrics"]["train_acc"][0])
    x_axis = np.arange(1, epochs + 1)

    # 1) ACCURACY CURVES ------------------------------------------------------
    try:
        plt.figure()
        for cfg, tr, va in zip(
            cfgs, ed["metrics"]["train_acc"], ed["metrics"]["val_acc"]
        ):
            plt.plot(x_axis, tr, label=f"{cfg}-train", alpha=0.8)
            plt.plot(x_axis, va, "--", label=f"{cfg}-val", alpha=0.8)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Train vs Validation Accuracy per Config")
        plt.legend(fontsize=8)
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 2) LOSS CURVES ----------------------------------------------------------
    try:
        plt.figure()
        for cfg, tr, va in zip(cfgs, ed["losses"]["train"], ed["losses"]["val"]):
            plt.plot(x_axis, tr, label=f"{cfg}-train", alpha=0.8)
            plt.plot(x_axis, va, "--", label=f"{cfg}-val", alpha=0.8)
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss per Config")
        plt.legend(fontsize=8)
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_loss_curves.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 3) RULE FIDELITY CURVES -------------------------------------------------
    try:
        plt.figure()
        for cfg, rf in zip(cfgs, ed["metrics"]["rule_fidelity"]):
            plt.plot(x_axis, rf, label=cfg, alpha=0.8)
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title("SPR_BENCH: Rule Fidelity over Epochs")
        plt.legend(fontsize=8)
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_rule_fidelity.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        print(f"Error creating rule fidelity plot: {e}")
        plt.close()

    # 4) CONFUSION MATRIX -----------------------------------------------------
    try:
        gt = ed["ground_truth"]
        pred = ed["predictions"]
        n_cls = len(np.unique(gt))
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gt, pred):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix (Best Config)")
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="black", fontsize=6
                )
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------------------------------------------------------------------
    print(
        f"Best configuration: {ed['best_config']}  "
        f"Dev accuracy (final epoch): {ed['metrics']['val_acc'][cfgs.index(ed['best_config'])][-1]:.3f}"
    )
