import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ------------------------------- paths -------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------- load experiment -------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

keys = list(experiment_data.keys())
colors = plt.cm.tab10.colors if keys else []


# ------------------------ helper: safe get ---------------------------
def g(d, *path, default=None):
    for p in path:
        if d is None:
            return default
        d = d.get(p, None)
    return d if d is not None else default


# -------------------- 1) LM pretrain loss curves ---------------------
for idx, k in enumerate(keys):
    try:
        pre_ls = g(experiment_data[k], "losses", "pretrain", default=[])
        if not pre_ls:
            continue
        plt.figure()
        epochs = np.arange(1, len(pre_ls) + 1)
        plt.plot(epochs, pre_ls, color=colors[idx % len(colors)])
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH – {k} – LM Pre-training Loss")
        fname = os.path.join(working_dir, f"{k}_lm_pretrain_loss.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting LM pretrain for {k}: {e}")
        plt.close()

# ---------------- 2) Train / Val loss & macro-F1 curves --------------
for idx, k in enumerate(keys):
    epochs = g(experiment_data[k], "epochs", default=[])
    if not epochs:
        continue
    c = colors[idx % len(colors)]

    # Loss curves
    try:
        tr_loss = g(experiment_data[k], "losses", "train", default=[])
        val_loss = g(experiment_data[k], "losses", "val", default=[])
        plt.figure()
        plt.plot(epochs, tr_loss, "--", color=c, label="train")
        plt.plot(epochs, val_loss, "-", color=c, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            f"SPR_BENCH – {k} – Classification Loss Curves\nLeft: Train (dashed)  Right: Val (solid)"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{k}_cls_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss curves for {k}: {e}")
        plt.close()

    # Macro-F1 curves
    try:
        tr_f1 = g(experiment_data[k], "metrics", "train_macro_f1", default=[])
        val_f1 = g(experiment_data[k], "metrics", "val_macro_f1", default=[])
        plt.figure()
        plt.plot(epochs, tr_f1, "--", color=c, label="train")
        plt.plot(epochs, val_f1, "-", color=c, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(
            f"SPR_BENCH – {k} – Macro-F1 Curves\nLeft: Train (dashed)  Right: Val (solid)"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{k}_macro_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting f1 curves for {k}: {e}")
        plt.close()

# --------------------- 3) Test Macro-F1 bar chart --------------------
try:
    test_scores = {
        k: g(experiment_data[k], "test_macro_f1", default=np.nan) for k in keys
    }
    plt.figure()
    plt.bar(
        range(len(test_scores)),
        list(test_scores.values()),
        tick_label=list(test_scores.keys()),
    )
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH – Test Macro-F1 per Setting")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_test_macro_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test score bar plot: {e}")
    plt.close()

# -------------------- 4) Confusion matrix heatmaps -------------------
max_conf_plots = 5
for idx, k in enumerate(keys[:max_conf_plots]):
    try:
        preds = np.array(g(experiment_data[k], "predictions", default=[]))
        gts = np.array(g(experiment_data[k], "ground_truth", default=[]))
        if preds.size == 0 or gts.size == 0:
            continue
        cm = confusion_matrix(gts, preds)
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"SPR_BENCH – {k} – Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{k}_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {k}: {e}")
        plt.close()

# ------------------------ 5) Numeric summary -------------------------
for k in keys:
    tl = g(experiment_data[k], "test_loss", default=None)
    tf1 = g(experiment_data[k], "test_macro_f1", default=None)
    print(f"{k:>20s} | Test loss: {tl:.4f} | Test Macro-F1: {tf1:.4f}")
