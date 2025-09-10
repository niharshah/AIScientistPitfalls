import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load experiment data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_runs = experiment_data.get("hidden_size", {}).get("SPR_BENCH", {})
hidden_sizes = sorted(spr_runs.keys())


# quick helpers
def to_np(seq):
    return np.array(seq)


# ------------------ figure 1: loss curves ------------------
try:
    plt.figure(figsize=(6, 4))
    for hs in hidden_sizes:
        epochs = np.arange(1, len(spr_runs[hs]["losses"]["train"]) + 1)
        plt.plot(
            epochs,
            to_np(spr_runs[hs]["losses"]["train"]),
            label=f"train_h{hs}",
            linestyle="-",
        )
        plt.plot(
            epochs,
            to_np(spr_runs[hs]["losses"]["val"]),
            label=f"val_h{hs}",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------ figure 2: F1 curves ------------------
try:
    plt.figure(figsize=(6, 4))
    for hs in hidden_sizes:
        epochs = np.arange(1, len(spr_runs[hs]["metrics"]["train"]) + 1)
        plt.plot(
            epochs,
            to_np(spr_runs[hs]["metrics"]["train"]),
            label=f"train_h{hs}",
            linestyle="-",
        )
        plt.plot(
            epochs,
            to_np(spr_runs[hs]["metrics"]["val"]),
            label=f"val_h{hs}",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH F1 Curves\nLeft: Train, Right: Validation")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_F1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# identify best run (highest final val F1)
best_hs, best_f1 = None, -1.0
for hs in hidden_sizes:
    f1 = spr_runs[hs]["metrics"]["val"][-1]
    if f1 > best_f1:
        best_f1, best_hs = f1, hs
print(f"Best hidden size: {best_hs} | Val F1: {best_f1:.3f}")

# ------------------ figure 3: final F1 bar chart ------------------
try:
    final_f1s = [spr_runs[hs]["metrics"]["val"][-1] for hs in hidden_sizes]
    plt.figure(figsize=(6, 4))
    plt.bar([str(hs) for hs in hidden_sizes], final_f1s, color="skyblue")
    plt.xlabel("Hidden Size")
    plt.ylabel("Final Val Macro F1")
    plt.title("SPR_BENCH Final Validation F1 by Hidden Size")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_val_F1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# ------------------ figure 4: confusion matrix for best model ------------------
try:
    preds = np.array(spr_runs[best_hs]["predictions"])
    gts = np.array(spr_runs[best_hs]["ground_truth"])
    cm = np.zeros((2, 2), dtype=int)
    for p, g in zip(preds, gts):
        cm[g, p] += 1

    plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"SPR_BENCH Confusion Matrix\nBest Hidden Size={best_hs}")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"SPR_BENCH_confusion_matrix_h{best_hs}.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
