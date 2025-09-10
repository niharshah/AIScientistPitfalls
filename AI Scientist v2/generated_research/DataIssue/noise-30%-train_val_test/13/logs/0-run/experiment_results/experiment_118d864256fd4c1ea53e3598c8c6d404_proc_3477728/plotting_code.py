import matplotlib.pyplot as plt
import numpy as np
import os

# set working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data -------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp_key, ds_key = "Frozen-Character-Embeddings", "SPR_BENCH"
ed = experiment_data.get(exp_key, {}).get(ds_key, {})

metrics = ed.get("metrics", {})
losses = ed.get("losses", {})
epochs = ed.get("epochs", [])

# print stored evaluation metrics --------------------------------------
if metrics:
    print(f"Test macro-F1: {metrics.get('test_f1'):.4f}")
    print(f"Systematic Generalization Accuracy: {metrics.get('SGA'):.4f}")

# 1. loss curve ---------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, losses.get("train", []), label="Train")
    plt.plot(epochs, losses.get("val", []), label="Validation")
    plt.title("SPR_BENCH Loss Curve (Frozen-Character-Embeddings)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve_FrozenCharEmb.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2. F1 curve -----------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, metrics.get("train_f1", []), label="Train F1")
    plt.plot(epochs, metrics.get("val_f1", []), label="Validation F1")
    plt.title("SPR_BENCH Macro-F1 Curve (Frozen-Character-Embeddings)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curve_FrozenCharEmb.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# 3. confusion matrix ---------------------------------------------------
try:
    from sklearn.metrics import confusion_matrix

    preds = np.array(ed.get("predictions", []))
    gts = np.array(ed.get("ground_truth", []))
    if preds.size and gts.size:
        cm = confusion_matrix(gts, preds, labels=sorted(set(gts)))
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("SPR_BENCH Confusion Matrix (Frozen-Character-Embeddings)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        fname = os.path.join(
            working_dir, "SPR_BENCH_confusion_matrix_FrozenCharEmb.png"
        )
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
