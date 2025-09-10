import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    vt_data = experiment_data["SPR_BENCH"]["variety_transformer"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    vt_data = None

if vt_data:
    tr_losses = vt_data["losses"]["train"]
    val_losses = vt_data["losses"]["val"]
    val_metrics = vt_data["metrics"]["val"]  # list of dicts per epoch
    test_metrics = vt_data["metrics"]["test"]  # dict
    preds = np.array(vt_data["predictions"])
    gts = np.array(vt_data["ground_truth"])
    epochs = np.arange(1, len(tr_losses) + 1)

    # -------- figure 1 : loss curves --------
    try:
        plt.figure()
        plt.plot(epochs, tr_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves (Variety-Transformer)")
        plt.legend()
        fname = "SPR_BENCH_variety_transformer_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------- figure 2 : validation metrics across epochs --------
    try:
        cwa = [m["CWA"] for m in val_metrics]
        swa = [m["SWA"] for m in val_metrics]
        gcw = [m["GCWA"] for m in val_metrics]
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, gcw, label="GCWA")
        plt.ylim(0, 1.01)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            "SPR_BENCH Validation Metrics vs Epoch\nLeft: CWA, Mid: SWA, Right: GCWA"
        )
        plt.legend()
        fname = "SPR_BENCH_variety_transformer_validation_metrics.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating validation metric plot: {e}")
        plt.close()

    # -------- figure 3 : bar chart of test metrics --------
    try:
        labels = list(test_metrics.keys())
        values = [test_metrics[k] for k in labels]
        plt.figure()
        plt.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Test Metrics (Variety-Transformer)")
        fname = "SPR_BENCH_variety_transformer_test_metrics_bar.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar plot: {e}")
        plt.close()

    # -------- figure 4 : confusion matrix --------
    try:
        n_cls = max(max(gts), max(preds)) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):  # rows=true, cols=pred
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title("SPR_BENCH Confusion Matrix (Variety-Transformer)")
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8,
                )
        fname = "SPR_BENCH_variety_transformer_confusion_matrix.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # -------- print test metrics --------
    print("Final SPR_BENCH test metrics (Variety-Transformer):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.3f}")
else:
    print("No valid SPR_BENCH Variety-Transformer data found.")
