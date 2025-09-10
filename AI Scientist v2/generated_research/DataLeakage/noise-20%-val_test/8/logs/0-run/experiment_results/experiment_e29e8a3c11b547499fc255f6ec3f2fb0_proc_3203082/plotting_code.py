import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    data_key = "criterion_tuning"
    ds_key = "SPR_BENCH"
    ed = experiment_data[data_key][ds_key]

    # Gather per-criterion stats
    crits, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
    for c, c_dict in ed["per_criterion"].items():
        crits.append(c)
        train_acc.append(c_dict["metrics"]["train"][0])
        val_acc.append(c_dict["metrics"]["val"][0])
        train_loss.append(c_dict["losses"]["train"][0])
        val_loss.append(c_dict["losses"]["val"][0])

    # -------------------- Plot 1: Accuracy bars --------------------
    try:
        x = np.arange(len(crits))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, train_acc, width, label="Train")
        plt.bar(x + width / 2, val_acc, width, label="Validation")
        plt.xticks(x, crits)
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Train vs Validation Accuracy per Criterion")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_per_criterion.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # -------------------- Plot 2: Loss bars ------------------------
    try:
        x = np.arange(len(crits))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, train_loss, width, label="Train")
        plt.bar(x + width / 2, val_loss, width, label="Validation")
        plt.xticks(x, crits)
        plt.ylabel("Loss (1-Acc)")
        plt.title("SPR_BENCH: Train vs Validation Loss per Criterion")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_per_criterion.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------------------- Plot 3: Predictions vs GT ---------------
    try:
        preds = np.array(ed["predictions"])
        gts = np.array(ed["ground_truth"])
        idx = np.arange(len(preds))
        plt.figure()
        plt.plot(idx, gts, label="Ground Truth", alpha=0.7)
        plt.plot(idx, preds, label="Predictions", alpha=0.7)
        plt.xlabel("Sample Index")
        plt.ylabel("Label")
        plt.title("SPR_BENCH: Ground Truth vs Predictions (Test Set)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_predictions_vs_gt.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating prediction plot: {e}")
        plt.close()

    # -------------------- Console metrics -------------------------
    best_crit = ed["best_criterion"]
    best_val = ed["metrics"]["val"][0]
    test_acc = np.mean(preds == gts)
    sefa = (
        np.round(ed.get("sefa", "N/A"), 4)
        if isinstance(ed.get("sefa", None), float)
        else "N/A"
    )
    print(f"Best criterion: {best_crit} | Best Val Acc: {best_val:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} | SEFA: {sefa}")
