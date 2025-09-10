import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------- load data ---------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["learning_rate"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

lrs = np.array(data["lrs"])
tr_losses = np.array(data["losses"]["train"])
val_losses = np.array(data["losses"]["val"])
val_mets = data["metrics"]["val"]  # list of dicts
test_mets = data["metrics"]["test"]  # list of dicts

# turn metric dict lists into arrays
val_cwa = np.array([m["CWA"] for m in val_mets])
val_swa = np.array([m["SWA"] for m in val_mets])
val_gcwa = np.array([m["GCWA"] for m in val_mets])
tst_cwa = np.array([m["CWA"] for m in test_mets])
tst_swa = np.array([m["SWA"] for m in test_mets])
tst_gcwa = np.array([m["GCWA"] for m in test_mets])

best_idx = int(np.argmax(val_gcwa))
best_lr = lrs[best_idx]
print("\n==== Summary of results on SPR_BENCH ====")
for i, lr in enumerate(lrs):
    print(
        f"LR={lr:.0e} | VAL CWA={val_cwa[i]:.3f} SWA={val_swa[i]:.3f} GCWA={val_gcwa[i]:.3f} "
        f"| TEST CWA={tst_cwa[i]:.3f} SWA={tst_swa[i]:.3f} GCWA={tst_gcwa[i]:.3f}"
    )
print(f"\nChosen best LR (highest VAL GCWA): {best_lr:.0e}\n")

# -------------- plotting ------------------ #
# 1) train vs val loss
try:
    plt.figure()
    w = 0.35
    idx = np.arange(len(lrs))
    plt.bar(idx - w / 2, tr_losses, width=w, label="Train Loss")
    plt.bar(idx + w / 2, val_losses, width=w, label="Val Loss")
    plt.xticks(idx, [f"{lr:.0e}" for lr in lrs])
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Final Train vs. Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_vs_lr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) validation metrics
try:
    plt.figure()
    plt.plot(lrs, val_cwa, "o-", label="CWA")
    plt.plot(lrs, val_swa, "s-", label="SWA")
    plt.plot(lrs, val_gcwa, "d-", label="GCWA")
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Validation Metrics vs. Learning Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_metrics_vs_lr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation metric plot: {e}")
    plt.close()

# 3) test metrics
try:
    plt.figure()
    plt.plot(lrs, tst_cwa, "o-", label="CWA")
    plt.plot(lrs, tst_swa, "s-", label="SWA")
    plt.plot(lrs, tst_gcwa, "d-", label="GCWA")
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Test Metrics vs. Learning Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics_vs_lr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test metric plot: {e}")
    plt.close()

# 4) confusion matrix for best LR
try:
    preds = np.array(data["predictions"][best_idx])
    tgts = np.array(data["ground_truth"][best_idx])
    classes = np.unique(tgts)
    n_cls = len(classes)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(tgts, preds):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"SPR_BENCH: Confusion Matrix (LR={best_lr:.0e})")
    plt.xticks(classes)
    plt.yticks(classes)
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
            )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"SPR_BENCH_confusion_matrix_lr_{best_lr:.0e}.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
