import os
import numpy as np

# -------------------------------------------------
# 0. Locate and load the saved experiment results
# -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(file_path, allow_pickle=True).item()


# -------------------------------------------------
# Helper functions (replicated from training code)
# -------------------------------------------------
def count_color_variety(sequence: str) -> int:
    return len({token[1] for token in sequence.strip().split() if len(token) > 1})


def count_shape_variety(sequence: str) -> int:
    return len({token[0] for token in sequence.strip().split() if token})


def complexity_weight(sequence: str) -> int:
    return count_color_variety(sequence) + count_shape_variety(sequence)


def complexity_weighted_accuracy(seqs, y_true, y_pred):
    weights = [complexity_weight(s) for s in seqs]
    correct_w = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return float(sum(correct_w)) / max(1, sum(weights))


# -------------------------------------------------
# 1. Iterate over datasets and print metrics
# -------------------------------------------------
for dataset_name, data in experiment_data.items():
    print(dataset_name)  # dataset header

    # ---- Final-epoch training metrics ----
    final_train_loss = data["losses"]["train"][-1]
    final_train_acc = data["metrics"]["train"][-1]["acc"]
    final_train_cowa = data["metrics"]["train"][-1]["cowa"]

    print(f"training loss: {final_train_loss:.4f}")
    print(f"training accuracy: {final_train_acc:.4f}")
    print(f"training complexity-weighted accuracy: {final_train_cowa:.4f}")

    # ---- Best-epoch validation metrics (chosen via highest val accuracy) ----
    val_metrics = data["metrics"]["val"]
    val_losses = data["losses"]["val"]
    best_idx = max(range(len(val_metrics)), key=lambda i: val_metrics[i]["acc"])

    best_val_loss = val_losses[best_idx]
    best_val_acc = val_metrics[best_idx]["acc"]
    best_val_cowa = val_metrics[best_idx]["cowa"]

    print(f"validation loss (best epoch): {best_val_loss:.4f}")
    print(f"validation accuracy (best epoch): {best_val_acc:.4f}")
    print(f"validation complexity-weighted accuracy (best epoch): {best_val_cowa:.4f}")

    # ---- Test metrics (computed from stored predictions & labels) ----
    preds = data.get("predictions", [])
    gts = data.get("ground_truth", [])
    seqs = data.get("sequences", [])

    if preds and gts and seqs:
        test_acc = sum(int(p == t) for p, t in zip(preds, gts)) / len(gts)
        test_cowa = complexity_weighted_accuracy(seqs, gts, preds)
        print(f"test accuracy: {test_acc:.4f}")
        print(f"test complexity-weighted accuracy: {test_cowa:.4f}")
