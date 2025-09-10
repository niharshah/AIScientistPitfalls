import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

key = "SPR_TransformerContrastive"
run = experiment_data.get(key, None)

if run is not None:
    losses_tr = run["losses"]["train"]
    losses_val = run["losses"]["val"]
    val_metrics = run["metrics"]["val"]  # list of dicts
    epochs = np.arange(1, len(losses_tr) + 1)
    # Extract metric arrays
    swa = [m.get("SWA", np.nan) for m in val_metrics]
    cwa = [m.get("CWA", np.nan) for m in val_metrics]
    comp = [m.get("CompWA", np.nan) for m in val_metrics]
    # Test arrays
    y_pred = np.array(run["predictions"])
    y_true = np.array(run["ground_truth"])
else:
    print("No run found.")
    losses_tr = losses_val = swa = cwa = comp = []
    epochs = np.array([])
    y_pred = y_true = np.array([])

# ------------------------------------------------------------------
# Plot 1: Train / Val Loss
try:
    if len(epochs):
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train Loss")
        plt.plot(epochs, losses_val, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH (synthetic) – Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Plot 2: Validation Weighted Accuracy curves
try:
    if len(epochs):
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, comp, label="CompWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH (synthetic) – Validation Weighted Accuracies")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_weighted_acc.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Plot 3: Confusion Matrix on Test Set
try:
    if y_true.size:
        from itertools import product

        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues", vmin=0)
        plt.colorbar()
        for i, j in product(range(2), repeat=2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title("SPR_BENCH (synthetic) – Test Confusion Matrix")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print evaluation metrics
if len(val_metrics):
    last = val_metrics[-1]
    print(
        f"Final-epoch Validation — SWA: {last.get('SWA', np.nan):.3f}, "
        f"CWA: {last.get('CWA', np.nan):.3f}, "
        f"CompWA: {last.get('CompWA', np.nan):.3f}"
    )
if y_true.size:
    # Re-compute test CompWA
    def complexity_weight(seq):
        return len(set(tok[0] for tok in seq.split())) + len(
            set(tok[1] for tok in seq.split())
        )

    seqs = run.get("ground_truth", [])  # actually sequences are not stored, skip
    # fall back to stored metric
    print(
        f"Test Complexity-Weighted Accuracy: "
        f"{run.get('test_comp', 'N/A') if isinstance(run, dict) else 'N/A'}"
    )
