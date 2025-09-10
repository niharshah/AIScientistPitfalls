import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

spr_data = experiment_data["LEARNING_RATE"]["SPR_BENCH"]
best_lr = spr_data["best_lr"]
best_rec = spr_data[f"{best_lr:.0e}"]


# ---------- helper for confused counts ----------
def confusion_counts(y_true, y_pred):
    tp = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0) and (yp == 0) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_true, y_pred))
    return np.array([[tn, fp], [fn, tp]])


# ---------- 1) Loss curves ----------
try:
    plt.figure()
    epochs = best_rec["epochs"]
    plt.plot(epochs, best_rec["losses"]["train"], label="Train")
    plt.plot(epochs, best_rec["losses"]["val"], label="Validation")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    fname = f"spr_bench_best_lr_{best_lr:.0e}_loss.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- 2) MCC curves ----------
try:
    plt.figure()
    plt.plot(epochs, best_rec["metrics"]["train_MCC"], label="Train")
    plt.plot(epochs, best_rec["metrics"]["val_MCC"], label="Validation")
    plt.title("SPR_BENCH MCC Curves\nLeft: Train, Right: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.legend()
    fname = f"spr_bench_best_lr_{best_lr:.0e}_mcc.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating MCC plot: {e}")
    plt.close()

# ---------- 3) LR sweep summary ----------
try:
    plt.figure()
    lrs = []
    peak_mcc = []
    for lr_key, rec in spr_data.items():
        if not lr_key.endswith("e"):  # skip aux keys
            continue
        lrs.append(lr_key)
        peak_mcc.append(max(rec["metrics"]["val_MCC"]))
    plt.bar(lrs, peak_mcc, color="skyblue")
    plt.title("SPR_BENCH Learning-Rate Sweep\nPeak Validation MCC per LR")
    plt.xlabel("Learning Rate")
    plt.ylabel("Peak Val MCC")
    plt.xticks(rotation=45)
    fname = "spr_bench_lr_sweep_mcc.png"
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating LR sweep plot: {e}")
    plt.close()

# ---------- 4) Confusion matrix ----------
try:
    y_true = spr_data["ground_truth"]
    y_pred = spr_data["predictions"]
    cm = confusion_counts(y_true, y_pred)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title(
        "SPR_BENCH Confusion Matrix on Test\nLeft: Ground Truth, Right: Predictions"
    )
    plt.xticks([0, 1], ["Neg", "Pos"])
    plt.yticks([0, 1], ["Neg", "Pos"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fname = "spr_bench_confusion_matrix.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- 5) Logit histogram ----------
try:
    # logits were not saved; reconstruct approximate logits from predictions as 0/1
    preds = np.array(y_pred)
    trues = np.array(y_true)
    plt.figure()
    plt.hist(preds[trues == 0], bins=2, alpha=0.7, label="True Negatives")
    plt.hist(preds[trues == 1], bins=2, alpha=0.7, label="True Positives")
    plt.title("SPR_BENCH Prediction Distribution\nLeft: True Neg, Right: True Pos")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.legend()
    fname = "spr_bench_pred_distribution.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating prediction distribution plot: {e}")
    plt.close()
