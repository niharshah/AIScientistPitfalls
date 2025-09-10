import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

wd_data = experiment_data.get("weight_decay_tuning", {})
weights = [
    k
    for k in wd_data.keys()
    if k not in {"best_wd", "predictions", "ground_truth", "test_mcc"}
]
best_wd = wd_data.get("best_wd", None)
test_mcc = wd_data.get("test_mcc", None)
print(f"Best weight_decay: {best_wd}, Test MCC: {test_mcc}")

# 1) Loss curves -----------------------------------------------------------
try:
    plt.figure()
    for w in weights:
        tr = wd_data[w]["losses"]["train"]
        vl = wd_data[w]["losses"]["val"]
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"train wd={w}")
        plt.plot(epochs, vl, "--", label=f"val wd={w}")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("SPR_BENCH Loss Curves for Different Weight Decays")
    plt.legend(fontsize=6)
    fpath = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) MCC curves ------------------------------------------------------------
try:
    plt.figure()
    for w in weights:
        tr = wd_data[w]["metrics"]["train"]
        vl = wd_data[w]["metrics"]["val"]
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"train wd={w}")
        plt.plot(epochs, vl, "--", label=f"val wd={w}")
    plt.xlabel("Epoch")
    plt.ylabel("Matthews Correlation Coef")
    plt.title("SPR_BENCH MCC Curves for Different Weight Decays")
    plt.legend(fontsize=6)
    fpath = os.path.join(working_dir, "SPR_BENCH_mcc_curves.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating MCC curves: {e}")
    plt.close()

# 3) Final val MCC bar chart ----------------------------------------------
try:
    plt.figure()
    final_vals = [wd_data[w]["metrics"]["val"][-1] for w in weights]
    plt.bar(
        weights,
        final_vals,
        color=["red" if w == str(best_wd) else "gray" for w in weights],
    )
    plt.xlabel("Weight Decay")
    plt.ylabel("Final Val MCC")
    plt.title(f"SPR_BENCH Final Validation MCC (best wd={best_wd})")
    plt.xticks(rotation=45)
    fpath = os.path.join(working_dir, "SPR_BENCH_val_mcc_bar.png")
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# 4) Confusion matrix heat-map --------------------------------------------
try:
    from sklearn.metrics import confusion_matrix

    preds = wd_data.get("predictions", None)
    gts = wd_data.get("ground_truth", None)
    if preds is not None and gts is not None:
        cm = confusion_matrix(gts, preds, labels=[0, 1])
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.title(
            "SPR_BENCH Test Confusion Matrix\nLeft: Ground Truth 0/1, Right: Predicted 0/1"
        )
        plt.colorbar(im)
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        fpath = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fpath, dpi=150)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
