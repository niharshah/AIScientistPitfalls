import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- setup & load ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# small helpers -------------------------------------------------
def _safe_list(d, *keys):
    cur = d
    for k in keys:
        cur = cur.get(k, [])
    return cur


# ----------------- FIGURE 1: contrastive loss -----------------
try:
    losses = [
        e["loss"] for e in _safe_list(experiment_data, "contrastive_pretrain", "loss")
    ]
    if losses:
        epochs = range(1, len(losses) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, losses, marker="o")
        plt.title("SPR_BENCH Contrastive Pre-train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_contrastive_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating contrastive loss plot: {e}")
    plt.close()

# -------- FIGURE 2: supervised train vs val loss ---------------
try:
    tr_loss = _safe_list(experiment_data, "supervised_finetune", "losses", "train")
    val_loss = _safe_list(experiment_data, "supervised_finetune", "losses", "val")
    if tr_loss and val_loss:
        epochs = range(1, len(tr_loss) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.title("SPR_BENCH Supervised Fine-tune Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_finetune_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating finetune loss plot: {e}")
    plt.close()

# ----------------- FIGURE 3: CCWA curve ------------------------
try:
    ccwa = [
        m["ccwa"]
        for m in _safe_list(experiment_data, "supervised_finetune", "metrics", "val")
    ]
    if ccwa:
        epochs = range(1, len(ccwa) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, ccwa, marker="s", color="green")
        plt.title("SPR_BENCH Validation CCWA Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("CCWA")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_CCWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CCWA plot: {e}")
    plt.close()

# ----------------- FIGURE 4: confusion matrix -----------------
try:
    preds = _safe_list(experiment_data, "supervised_finetune", "predictions")
    gts = _safe_list(experiment_data, "supervised_finetune", "ground_truth")
    if preds and gts:
        n_lbl = max(max(preds), max(gts)) + 1
        cm = np.zeros((n_lbl, n_lbl), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure(figsize=(5, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title("SPR_BENCH Confusion Matrix\nLeft: True, Top: Predicted")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
