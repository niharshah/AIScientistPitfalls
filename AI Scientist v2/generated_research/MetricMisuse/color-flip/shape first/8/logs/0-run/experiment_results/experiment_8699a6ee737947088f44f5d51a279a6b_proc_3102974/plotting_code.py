import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

cfg = experiment_data.get("remove_positional_embeddings", {}).get("SPR_BENCH", {})
loss_train = cfg.get("losses", {}).get("train", [])
loss_val = cfg.get("losses", {}).get("val", [])
ccwa_val = cfg.get("metrics", {}).get("val_CCWA", [])
preds_all = cfg.get("predictions", [])
gts_all = cfg.get("ground_truth", [])

saved = []

# ------------- plot 1: loss curves -------------
try:
    plt.figure()
    epochs = range(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Loss Curves (No Positional Embeddings)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_no_pos_emb.png")
    plt.savefig(fname)
    saved.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------- plot 2: CCWA metric -------------
try:
    plt.figure()
    plt.plot(range(1, len(ccwa_val) + 1), ccwa_val, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("CCWA")
    plt.title("SPR_BENCH – Validation CCWA (No Positional Embeddings)")
    fname = os.path.join(working_dir, "SPR_BENCH_CCWA_no_pos_emb.png")
    plt.savefig(fname)
    saved.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CCWA plot: {e}")
    plt.close()


# ------------- helper for last-epoch data -----
def latest(lst):
    return lst[-1] if isinstance(lst, list) and lst else []


y_pred = latest(preds_all)
y_true = latest(gts_all)

# ------------- plot 3: confusion matrix -------
try:
    if y_pred and y_true:
        import itertools

        classes = sorted(set(y_true) | set(y_pred))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[classes.index(t), classes.index(p)] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(ticks=range(len(classes)), labels=classes)
        plt.yticks(ticks=range(len(classes)), labels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH – Confusion Matrix (Last Epoch)")
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix_last_epoch.png")
        plt.savefig(fname)
        saved.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------- plot 4: class distribution -----
try:
    if y_pred and y_true:
        classes = sorted(set(y_true) | set(y_pred))
        true_counts = [y_true.count(c) for c in classes]
        pred_counts = [y_pred.count(c) for c in classes]
        x = np.arange(len(classes))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, true_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_counts, width, label="Predicted")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("SPR_BENCH – Class Distribution (Last Epoch)")
        plt.xticks(x, classes)
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_class_distribution_last_epoch.png")
        plt.savefig(fname)
        saved.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating class distribution plot: {e}")
    plt.close()

print("Saved figures:")
for f in saved:
    print(" -", f)
