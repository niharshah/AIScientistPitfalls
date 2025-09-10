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
    experiment_data = {}

run_key = "remove_final_layernorm"
dset_key = "SPR_BENCH"
run = experiment_data.get(run_key, {}).get(dset_key, {})

# ---------- Plot 1: Train/Val loss ----------
try:
    losses = run.get("losses", {})
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    epochs = range(1, min(len(train_loss), len(val_loss)) + 1)

    plt.figure()
    plt.plot(epochs, train_loss[: len(epochs)], label="Train Loss")
    plt.plot(epochs, val_loss[: len(epochs)], label="Val Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(
        working_dir, "SPR_BENCH_loss_curves_remove_final_layernorm.png"
    )
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- Plot 2: Val Macro-F1 & CWA ----------
try:
    metrics = run.get("metrics", {}).get("val", [])
    macro_f1 = [m.get("macro_f1") for m in metrics if m]
    cwa = [m.get("cwa") for m in metrics if m]
    epochs = range(1, len(macro_f1) + 1)

    plt.figure()
    ax1 = plt.gca()
    ax1.plot(epochs, macro_f1, "b-", label="Macro-F1")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Macro-F1", color="b")
    ax2 = ax1.twinx()
    ax2.plot(epochs, cwa, "r--", label="CWA")
    ax2.set_ylabel("CWA", color="r")
    plt.title("SPR_BENCH Validation Metrics\nLeft: Macro-F1, Right: CWA")
    fname = os.path.join(
        working_dir, "SPR_BENCH_val_metrics_remove_final_layernorm.png"
    )
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# ---------- Plot 3: Confusion Matrix ----------
try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    preds = np.array(run.get("predictions", []))
    gts = np.array(run.get("ground_truth", []))
    if preds.size and gts.size:
        cm = confusion_matrix(gts, preds)
        plt.figure()
        disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
        disp.plot(cmap="Blues", ax=plt.gca(), colorbar=False)
        plt.title("SPR_BENCH Confusion Matrix\nFinal Dev Predictions")
        fname = os.path.join(
            working_dir, "SPR_BENCH_confusion_matrix_remove_final_layernorm.png"
        )
        plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
