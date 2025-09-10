import os
import numpy as np


# ----------------------------------------------------------------------
# Helpers (replicate small parts of training code so we can recompute CWA)
# ----------------------------------------------------------------------
def colour_of(token: str) -> str:
    return token[1:] if len(token) > 1 else ""


def shape_of(token: str) -> str:
    return token[0]


def count_colour_variety(seq: str) -> int:
    return len(set(colour_of(tok) for tok in seq.split() if tok))


def count_shape_variety(seq: str) -> int:
    return len(set(shape_of(tok) for tok in seq.split() if tok))


def complexity_weight(seq: str) -> int:
    return count_colour_variety(seq) + count_shape_variety(seq)


def comp_weighted_acc(seqs, y_true, y_pred):
    weights = [complexity_weight(s) for s in seqs]
    correct_weights = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct_weights) / max(1, sum(weights))


# ----------------------------------------------------------------------
# Load experiment data
# ----------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
exp_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(exp_path, allow_pickle=True).item()

# ----------------------------------------------------------------------
# Iterate over datasets and print requested metrics
# ----------------------------------------------------------------------
for dataset_name, data in experiment_data.items():
    print(dataset_name)  # dataset header

    # --------------------- training metrics (final epoch) ---------------------
    final_train_metrics = data["metrics"]["train"][-1]
    final_train_loss = data["losses"]["train"][-1]
    print(f"training accuracy: {final_train_metrics['acc']:.3f}")
    print(f"training loss: {final_train_loss:.4f}")

    # --------------------- validation metrics (best epoch) --------------------
    best_epoch = data.get("best_epoch", final_train_metrics["epoch"])
    # locate best-epoch entry
    val_metrics_list = data["metrics"]["val"]
    best_val_entry = next(m for m in val_metrics_list if m["epoch"] == best_epoch)
    best_val_loss = data["losses"]["val"][best_epoch - 1]  # epoch indices start at 1
    print(f"validation accuracy (best epoch): {best_val_entry['acc']:.3f}")
    print(f"validation loss (best epoch): {best_val_loss:.4f}")
    print(f"validation CompWA (best epoch): {best_val_entry['CompWA']:.3f}")

    # --------------------- test metrics (re-computed) -------------------------
    preds = data.get("predictions", [])
    gts = data.get("ground_truth", [])
    seqs = data.get("sequences", [])

    if preds and gts and seqs:
        test_accuracy = sum(p == g for p, g in zip(preds, gts)) / len(gts)
        test_compwa = comp_weighted_acc(seqs, gts, preds)
        print(f"test accuracy: {test_accuracy:.3f}")
        print(f"test CompWA: {test_compwa:.3f}")
