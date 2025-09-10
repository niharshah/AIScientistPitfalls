import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr is not None:

    # ---------------- plot 1: loss curves ----------------
    try:
        plt.figure()
        epochs = np.arange(1, len(spr["losses"]["train"]) + 1)
        plt.plot(epochs, spr["losses"]["train"], label="Train")
        plt.plot(epochs, spr["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Loss vs Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------------- plot 2: macro-F1 curves -------------
    try:
        plt.figure()
        epochs = np.arange(1, len(spr["metrics"]["train"]) + 1)
        plt.plot(epochs, spr["metrics"]["train"], label="Train")
        plt.plot(epochs, spr["metrics"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH – Macro-F1 vs Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # ---------------- plot 3: test macro-F1 summary -------
    try:
        plt.figure()
        hp_scores = {k: v["test_macro_f1"] for k, v in spr["hyperparams"].items()}
        keys, vals = list(hp_scores.keys()), list(hp_scores.values())
        plt.bar(range(len(vals)), vals, tick_label=[str(i) for i in range(len(vals))])
        plt.ylabel("Test Macro-F1")
        plt.xlabel("Hyper-parameter Index")
        plt.title("SPR_BENCH – Test Macro-F1 per Hyper-param")
        for idx, v in enumerate(vals):
            plt.text(idx, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        fname = os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()

    # ---------------- plot 4: confusion matrix ------------
    try:
        best_hp = max(hp_scores, key=hp_scores.get)
        best_idx = list(spr["hyperparams"].keys()).index(best_hp)
        preds = np.array(spr["predictions"][best_idx])
        gts = np.array(spr["ground_truth"][best_idx])
        labels = np.unique(gts)
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for gt, pr in zip(gts, preds):
            cm[np.where(labels == gt)[0][0], np.where(labels == pr)[0][0]] += 1
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        plt.figure()
        im = plt.imshow(cm_norm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title(f"SPR_BENCH – Confusion Matrix (Best HP {best_hp})")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------- print evaluation metrics ---------------
    print("Test Macro-F1 per hyper-parameter:")
    print(hp_scores)
