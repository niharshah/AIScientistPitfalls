import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

variants = ["Baseline", "SymbolicAug"]
colors = {"Baseline": "tab:blue", "SymbolicAug": "tab:orange"}


# ---------- helper ----------
def get_arr(v, path):
    try:
        arr = experiment_data[v]
        for key in path:
            arr = arr[key]
        return np.array(arr)
    except KeyError:
        return None


# ---------- 1) train loss ----------
try:
    plt.figure()
    plotted = False
    for v in variants:
        epochs = get_arr(v, ["epochs"])
        tr_loss = get_arr(v, ["losses", "train"])
        if epochs is None or tr_loss is None:
            continue
        plt.plot(epochs, tr_loss, label=v, color=colors[v])
        plotted = True
    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH – Training Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_train_loss_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating train-loss plot: {e}")
    plt.close()

# ---------- 2) validation loss ----------
try:
    plt.figure()
    plotted = False
    for v in variants:
        epochs = get_arr(v, ["epochs"])
        val_loss = get_arr(v, ["losses", "val"])
        if epochs is None or val_loss is None:
            continue
        plt.plot(epochs, val_loss, label=v, color=colors[v])
        plotted = True
    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH – Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_val_loss_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val-loss plot: {e}")
    plt.close()

# ---------- 3) validation macro-F1 ----------
try:
    plt.figure()
    plotted = False
    for v in variants:
        epochs = get_arr(v, ["epochs"])
        val_f1 = get_arr(v, ["metrics", "val_f1"])
        if epochs is None or val_f1 is None:
            continue
        plt.plot(epochs, val_f1, label=v, color=colors[v])
        plotted = True
    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("SPR_BENCH – Validation Macro-F1")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_val_f1_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val-F1 plot: {e}")
    plt.close()


# ---------- 4 & 5) confusion matrices ----------
def confusion_matrix(preds, gts, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(preds, gts):
        cm[t, p] += 1
    return cm


for v in variants:
    try:
        preds = get_arr(v, ["predictions"])
        gts = get_arr(v, ["ground_truth"])
        if preds is None or gts is None or len(preds) == 0:
            continue
        num_classes = len(set(gts))
        cm = confusion_matrix(preds, gts, num_classes)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"SPR_BENCH – Confusion Matrix ({v})")
        plt.savefig(os.path.join(working_dir, f"spr_confusion_matrix_{v.lower()}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {v}: {e}")
        plt.close()

# ---------- numeric summary ----------
for v in variants:
    val_f1 = get_arr(v, ["metrics", "val_f1"])
    if val_f1 is not None and val_f1.size:
        print(f"{v}: best val Macro-F1 = {val_f1.max():.4f}")
