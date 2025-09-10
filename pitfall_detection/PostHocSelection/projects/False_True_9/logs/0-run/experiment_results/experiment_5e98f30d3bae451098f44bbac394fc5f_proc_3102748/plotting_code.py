import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- iterate datasets ----------
for ds, rec in experiment_data.items():
    losses = rec.get("losses", {})
    metrics = rec.get("metrics", {})
    preds = rec.get("predictions", [])
    gts = rec.get("ground_truth", [])

    # ---- 1. pre-training loss ----
    try:
        pre = losses.get("pretrain", [])
        if pre:
            ep, val = zip(*pre)
            plt.figure()
            plt.plot(ep, val, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("InfoNCE Loss")
            plt.title(f"{ds}: Pre-Training Loss Curve")
            fname = os.path.join(working_dir, f"{ds}_pretrain_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating pretrain plot for {ds}: {e}")
        plt.close()

    # ---- 2. train vs val loss ----
    try:
        tr = losses.get("train", [])
        vl = losses.get("val", [])
        if tr and vl:
            ep_t, v_t = zip(*tr)
            ep_v, v_v = zip(*vl)
            plt.figure()
            plt.plot(ep_t, v_t, label="Train")
            plt.plot(ep_v, v_v, linestyle="--", label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds}: Fine-Tuning Loss (Train vs Val)")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_train_val_loss.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating train/val loss plot for {ds}: {e}")
        plt.close()

    # ---- 3. validation metrics ----
    try:
        val_m = metrics.get("val", [])
        if val_m:
            ep, swa, cwa, dawa = zip(*val_m)
            plt.figure()
            plt.plot(ep, swa, label="SWA")
            plt.plot(ep, cwa, label="CWA")
            plt.plot(ep, dawa, label="DAWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{ds}: Validation Metrics Across Epochs")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_val_metrics.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {ds}: {e}")
        plt.close()

    # ---- 4. confusion matrix ----
    try:
        if preds and gts:
            classes = sorted(set(gts) | set(preds))
            n = len(classes)
            cm = np.zeros((n, n), dtype=int)
            cls2idx = {c: i for i, c in enumerate(classes)}
            for y, p in zip(gts, preds):
                cm[cls2idx[y], cls2idx[p]] += 1
            plt.figure(figsize=(4, 4))
            im = plt.imshow(cm, cmap="Blues")
            plt.title(f"{ds}: Confusion Matrix (Dev Set)")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.colorbar(im, fraction=0.046)
            plt.xticks(range(n), classes, rotation=90)
            plt.yticks(range(n), classes)
            plt.tight_layout()
            fname = os.path.join(working_dir, f"{ds}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds}: {e}")
        plt.close()

    # ---- print final metrics ----
    if metrics.get("val"):
        last_ep, last_s, last_c, last_d = metrics["val"][-1]
        acc = (
            np.mean(np.array(preds) == np.array(gts)) if preds and gts else float("nan")
        )
        print(
            f"{ds} – Final Epoch {last_ep}: DAWA={last_d:.4f}, "
            f"SWA={last_s:.4f}, CWA={last_c:.4f}, Acc={acc:.4f}"
        )
