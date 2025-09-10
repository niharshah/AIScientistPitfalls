import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to get curves
def get_curves(ds, key_outer, key_inner):
    outer = experiment_data[ds][key_outer]
    return (
        {d: np.array(outer[d]) for d in outer}
        if key_inner is None
        else {d: np.array(outer[d][key_inner]) for d in outer}
    )


dataset = "SPR_BENCH"
if dataset in experiment_data:
    # ---------- Plot 1: Loss curves ----------
    try:
        plt.figure()
        # train and val curves per dropout
        train_curves = get_curves(dataset, "losses", "train")
        val_curves = get_curves(dataset, "losses", "val")
        for d in train_curves:
            plt.plot(train_curves[d], label=f"dropout {d} - train")
            plt.plot(val_curves[d], linestyle="--", label=f"dropout {d} - val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train/Val Loss vs Epochs")
        plt.legend(fontsize=8)
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- Plot 2: F1 curves ----------
    try:
        plt.figure()
        train_f1 = get_curves(dataset, "metrics", "train_f1")
        val_f1 = get_curves(dataset, "metrics", "val_f1")
        for d in train_f1:
            plt.plot(train_f1[d], label=f"dropout {d} - train")
            plt.plot(val_f1[d], linestyle="--", label=f"dropout {d} - val")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Train/Val Macro-F1 vs Epochs")
        plt.legend(fontsize=8)
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve plot: {e}")
        plt.close()

    # ---------- Plot 3: Test F1 bar ----------
    try:
        plt.figure()
        test_f1 = experiment_data[dataset]["metrics"]["test_f1"]
        drops, scores = zip(*sorted(test_f1.items()))
        plt.bar(range(len(scores)), scores, tick_label=[str(d) for d in drops])
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Test Macro-F1 vs Dropout")
        fname = os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Test-F1 bar plot: {e}")
        plt.close()

    # ---------- Plot 4-6: Confusion matrices (max 3) ----------
    try:
        preds_dict = experiment_data[dataset].get("predictions", {})
        gt_dict = experiment_data[dataset].get("ground_truth", {})
        plotted = 0
        for d in sorted(preds_dict.keys()):
            if plotted >= 3:
                break
            preds = np.array(preds_dict[d])
            gts = np.array(gt_dict[d])
            if preds.size == 0 or gts.size == 0:
                continue
            classes = np.unique(np.concatenate([preds, gts]))
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for p, g in zip(preds, gts):
                cm[g, p] += 1
            plt.figure(figsize=(4, 3))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"SPR_BENCH Confusion Matrix (dropout={d})")
            plt.xticks(classes)
            plt.yticks(classes)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=6,
                    )
            fname = os.path.join(working_dir, f"SPR_BENCH_confusion_dropout_{d}.png")
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
            plotted += 1
    except Exception as e:
        print(f"Error creating confusion matrix plots: {e}")
        plt.close()
