import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data ---------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

variants = ["ORIG_SYM", "SHUFFLED_SYM"]
dataset = "SPR_BENCH"


# -------- helper for confusion matrix ---
def conf_mat(y_true, y_pred, n_cls):
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# -------------------- PLOT 1 : losses ---------------------------
try:
    plt.figure(figsize=(6, 4))
    for var in variants:
        rec = experiment_data[var][dataset]
        epochs = np.arange(1, len(rec["losses"]["train"]) + 1)
        plt.plot(epochs, rec["losses"]["train"], label=f"{var}-train")
        plt.plot(epochs, rec["losses"]["val"], label=f"{var}-val", linestyle="--")
    plt.title("SPR_BENCH Loss Curves\nTrain vs Validation (ORIG & SHUFFLED)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------------------- PLOT 2 : SWA ------------------------------
try:
    plt.figure(figsize=(6, 4))
    for var in variants:
        rec = experiment_data[var][dataset]
        epochs = np.arange(1, len(rec["metrics"]["train_swa"]) + 1)
        plt.plot(epochs, rec["metrics"]["train_swa"], label=f"{var}-train")
        plt.plot(epochs, rec["metrics"]["val_swa"], label=f"{var}-val", linestyle="--")
    plt.title(
        "SPR_BENCH Shape-Weighted Accuracy\nTrain vs Validation (ORIG & SHUFFLED)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_swa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# ---------------- Confusion matrices ---------------------------
for var in variants:
    try:
        rec = experiment_data[var][dataset]
        y_true = rec["ground_truth"]
        y_pred = rec["predictions"]
        if y_true is None or y_pred is None or len(y_true) == 0:
            raise ValueError("Missing prediction data")
        n_cls = int(max(y_true.max(), y_pred.max())) + 1
        cm = conf_mat(y_true, y_pred, n_cls)
        plt.figure(figsize=(5, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"Confusion Matrix - {var} (SPR_BENCH)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        fname = os.path.join(working_dir, f"spr_bench_conf_mat_{var.lower()}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion for {var}: {e}")
        plt.close()
