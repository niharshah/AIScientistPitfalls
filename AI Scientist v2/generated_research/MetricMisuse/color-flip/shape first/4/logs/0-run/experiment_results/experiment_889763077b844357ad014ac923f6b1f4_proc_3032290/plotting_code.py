import matplotlib.pyplot as plt
import numpy as np
import os

# setup paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data.get("SPR_Contrastive", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = {}

# ---------- Plot 1: Train / Val loss curves ----------
try:
    tr_loss = run.get("losses", {}).get("train", [])
    val_loss = run.get("losses", {}).get("val", [])
    if tr_loss and val_loss:
        x = np.arange(1, len(tr_loss) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(x, tr_loss, "--o", label="train")
        plt.plot(x, val_loss, "-s", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_Contrastive: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_Contrastive_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        print("Loss data missing; skipping loss curve plot.")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
finally:
    plt.close()

# ---------- Plot 2: Validation CoWA curves ----------
try:
    val_metrics = run.get("metrics", {}).get("val", [])
    cowa = [m.get("CoWA") for m in val_metrics if m and m.get("CoWA") is not None]
    if cowa:
        x = np.arange(1, len(cowa) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(x, cowa, "-^", label="CoWA")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title("SPR_Contrastive: Validation CoWA Across Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_Contrastive_CoWA_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        print("CoWA data missing; skipping CoWA plot.")
except Exception as e:
    print(f"Error creating CoWA plot: {e}")
finally:
    plt.close()

# ---------- Plot 3: Final epoch class distribution ----------
try:
    preds_all = run.get("predictions", [])
    gts_all = run.get("ground_truth", [])
    if preds_all and gts_all:
        y_pred = preds_all[-1]
        y_true = gts_all[-1]
        classes = sorted(set(y_true + y_pred))
        pred_counts = [y_pred.count(c) for c in classes]
        true_counts = [y_true.count(c) for c in classes]

        ind = np.arange(len(classes))
        width = 0.35
        plt.figure(figsize=(6, 4))
        plt.bar(ind, true_counts, width, label="Ground Truth")
        plt.bar(ind + width, pred_counts, width, label="Predictions")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("SPR_Contrastive: Class Distribution (Last Epoch)")
        plt.xticks(ind + width / 2, classes)
        plt.legend()
        fname = os.path.join(working_dir, "SPR_Contrastive_final_class_dist.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        print("Prediction data missing; skipping distribution plot.")
except Exception as e:
    print(f"Error creating class distribution plot: {e}")
finally:
    plt.close()

# ---------- Print final metrics ----------
try:
    if val_loss and cowa:
        print(f"Final Val Loss: {val_loss[-1]:.4f}")
        print(f"Final Val CoWA: {cowa[-1]:.3f}")
except Exception as e:
    print(f"Error printing final metrics: {e}")
