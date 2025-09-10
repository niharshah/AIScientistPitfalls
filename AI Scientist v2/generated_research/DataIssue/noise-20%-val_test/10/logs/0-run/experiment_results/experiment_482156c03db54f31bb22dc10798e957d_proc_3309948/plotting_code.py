import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------#
#  load experiment data
# ------------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------#
#  select run & pull arrays
# ------------------------------------------------------------------#
run = experiment_data.get("freeze_char_emb", {}).get("SPR_BENCH", {})
loss_tr = run.get("losses", {}).get("train", [])
loss_val = run.get("losses", {}).get("val", [])
f1_tr = run.get("metrics", {}).get("train_f1", [])
f1_val = run.get("metrics", {}).get("val_f1", [])
preds_test = np.array(run.get("preds_test", []))
gts_test = np.array(run.get("gts_test", []))

# ------------------------------------------------------------------#
#  print stored metrics
# ------------------------------------------------------------------#
print("Best val F1:", max(f1_val) if f1_val else None)
print("Rule-Extraction Acc dev:", run.get("metrics", {}).get("REA_dev"))
print("Rule-Extraction Acc test:", run.get("metrics", {}).get("REA_test"))
print(
    "Hybrid Model Test Macro-F1:",
    run.get("metrics", {}).get("val_f1")[-1] if f1_val else None,
)

# ------------------------------------------------------------------#
#  Figure 1: Loss curves
# ------------------------------------------------------------------#
try:
    plt.figure()
    epochs = range(1, len(loss_tr) + 1)
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Train vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------#
#  Figure 2: F1 curves
# ------------------------------------------------------------------#
try:
    plt.figure()
    plt.plot(epochs, f1_tr, label="Train")
    plt.plot(epochs, f1_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH – Train vs Validation Macro-F1")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# ------------------------------------------------------------------#
#  Figure 3: Confusion matrix on test split
# ------------------------------------------------------------------#
try:
    if preds_test.size and gts_test.size:
        num_classes = len(set(gts_test))
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(gts_test, preds_test):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH – Confusion Matrix (Test)")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
