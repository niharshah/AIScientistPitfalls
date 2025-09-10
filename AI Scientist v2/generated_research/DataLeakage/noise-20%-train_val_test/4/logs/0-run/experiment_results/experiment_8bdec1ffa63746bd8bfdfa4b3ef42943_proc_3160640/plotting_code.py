import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    spr_exp = experiment_data["weight_decay"]["SPR_BENCH"]
    wd_keys = [k for k in spr_exp.keys() if is_float(k)]  # numeric keys only

    # 1) Train / Val F1 curves
    try:
        plt.figure()
        for k in wd_keys:
            epochs = spr_exp[k]["epochs"]
            plt.plot(
                epochs, spr_exp[k]["metrics"]["train_f1"], "--", label=f"train wd={k}"
            )
            plt.plot(epochs, spr_exp[k]["metrics"]["val_f1"], "-", label=f"val wd={k}")
        plt.title(
            "SPR_BENCH: Train vs Validation F1 across epochs\n(hyper-parameter sweep)"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.legend(fontsize=8)
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve plot: {e}")
        plt.close()

    # 2) Best Val F1 per weight decay
    try:
        best_vals = [max(spr_exp[k]["metrics"]["val_f1"]) for k in wd_keys]
        plt.figure()
        plt.bar(range(len(wd_keys)), best_vals, tick_label=wd_keys)
        plt.title("SPR_BENCH: Best Validation F1 vs Weight Decay")
        plt.xlabel("Weight Decay")
        plt.ylabel("Best Val Macro F1")
        fname = os.path.join(working_dir, "SPR_BENCH_best_val_f1_bar.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating best-val-F1 bar plot: {e}")
        plt.close()

    # 3) Confusion matrix for best model
    try:
        preds = (
            np.array(spr_exp["predictions"])
            if "predictions" in spr_exp
            else np.array([])
        )
        gts = (
            np.array(spr_exp["ground_truth"])
            if "ground_truth" in spr_exp
            else np.array([])
        )
        if preds.size and gts.size:
            n_labels = int(max(gts.max(), preds.max()) + 1)
            conf = np.zeros((n_labels, n_labels), dtype=int)
            for gt, pr in zip(gts, preds):
                conf[gt, pr] += 1
            plt.figure()
            im = plt.imshow(conf, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(
                "SPR_BENCH: Confusion Matrix on Test Set\n(Left axis: Ground Truth, Bottom axis: Predictions)"
            )
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close()
        else:
            print("Predictions / ground truth not found, skipping confusion matrix.")
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()
