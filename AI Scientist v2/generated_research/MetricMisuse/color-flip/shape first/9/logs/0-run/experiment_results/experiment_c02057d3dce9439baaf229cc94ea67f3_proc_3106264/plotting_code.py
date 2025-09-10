import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

# helper to extract run dict safely
run = None
if experiment_data is not None:
    try:
        run = experiment_data["MaskedLanguageModelingPretrain"]["SPR_BENCH"]
    except KeyError:
        print("Run SPR_BENCH not found inside experiment data.")

# ---------- plot 1: loss curves ----------
try:
    if run is None:  # skip if data missing
        raise ValueError("No run dictionary available.")
    tr_loss = run["losses"]["train"]
    val_loss = run["losses"]["val"]
    epochs = np.arange(1, len(val_loss) + 1)

    plt.figure()
    if tr_loss:
        plt.plot(epochs, tr_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- plot 2: validation metric curve ----------
try:
    if run is None:
        raise ValueError("No run dictionary available.")
    val_metric = run["metrics"]["val"]
    if not val_metric or all(v is None for v in val_metric):
        raise ValueError("No validation metric data.")
    epochs = np.arange(1, len(val_metric) + 1)
    plt.figure()
    plt.plot(epochs, val_metric, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Comp-Weighted Accuracy")
    plt.title("SPR_BENCH Validation Comp-WA\nDataset: SPR_BENCH")
    fname = os.path.join(working_dir, "SPR_BENCH_CWA_curve.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating metric curve plot: {e}")
    plt.close()

# ---------- plot 3: confusion matrix ----------
try:
    if run is None:
        raise ValueError("No run dictionary available.")
    preds = run.get("predictions", [])
    gts = run.get("ground_truth", [])
    if len(preds) == 0 or len(preds) != len(gts):
        raise ValueError("Predictions/Ground truth arrays invalid.")
    num_classes = max(max(preds), max(gts)) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- print best metric ----------
if run is not None and run["metrics"]["val"]:
    best_cwa = max([m for m in run["metrics"]["val"] if m is not None])
    print(f"Best validation Comp-Weighted Accuracy: {best_cwa:.4f}")
