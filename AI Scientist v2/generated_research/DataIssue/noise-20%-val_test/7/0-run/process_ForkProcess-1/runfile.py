import os
import numpy as np

# ---------- LOAD SAVED EXPERIMENT DATA ----------
working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find experiment file at: {file_path}")

experiment_data = np.load(file_path, allow_pickle=True).item()


# ---------- HELPER TO FORMAT METRIC VALUES ----------
def fmt(val, precision=4):
    return f"{val:.{precision}f}" if isinstance(val, (float, np.floating)) else str(val)


# ---------- METRIC EXTRACTION ----------
for ds_name, ds_blob in experiment_data.items():
    print(f"\nDataset: {ds_name}")

    # --- Train metrics ---
    train_acc = ds_blob.get("metrics", {}).get("train_acc", [])
    if train_acc:
        print(f"final train accuracy: {fmt(train_acc[-1])}")

    train_losses = ds_blob.get("losses", {}).get("train", [])
    if train_losses:
        print(f"final train loss: {fmt(train_losses[-1])}")

    # --- Validation metrics ---
    val_acc = ds_blob.get("metrics", {}).get("val_acc", [])
    if val_acc:
        print(f"best validation accuracy: {fmt(max(val_acc))}")

    val_loss = ds_blob.get("metrics", {}).get("val_loss", [])
    if val_loss:
        print(f"best validation loss: {fmt(min(val_loss))}")

    # --- Test metrics (re-compute from stored predictions) ---
    preds = ds_blob.get("predictions")
    gts = ds_blob.get("ground_truth")
    if preds is not None and gts is not None and len(preds) == len(gts):
        test_accuracy = (preds == gts).mean()
        print(f"test accuracy: {fmt(test_accuracy)}")
    else:
        test_accuracy = None  # fallback if unavailable

    # --- Rule fidelity ---
    rule_preds = ds_blob.get("rule_preds")
    if rule_preds is not None and preds is not None and len(rule_preds) == len(preds):
        rule_fidelity = (rule_preds == preds).mean()
        print(f"rule fidelity: {fmt(rule_fidelity)}")
    else:
        rule_fidelity = None

    # --- FAGM (Fidelity–Accuracy Geometric Mean) ---
    if test_accuracy is not None and rule_fidelity is not None:
        fagm = np.sqrt(test_accuracy * rule_fidelity)
        print(f"FAGM: {fmt(fagm)}")
