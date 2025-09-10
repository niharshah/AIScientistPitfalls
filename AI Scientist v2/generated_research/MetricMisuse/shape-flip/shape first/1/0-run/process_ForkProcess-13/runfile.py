import os
import numpy as np

# ---------------------------------------------------------------------
# locate and load the experiment data ---------------------------------
working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(file_path, allow_pickle=True).item()

# ---------------------------------------------------------------------
# iterate over datasets and hyper-parameter settings ------------------
search_space = experiment_data.get("hidden_dim_tuning", {})
for dataset_name, runs in search_space.items():
    print(f"\nDataset: {dataset_name}")  # requirement (3)

    for run_name, mdata in runs.items():
        # retrieve recorded metric sequences
        train_accs = mdata["metrics"].get("train_acc", [])
        val_accs = mdata["metrics"].get("val_acc", [])
        val_losses = mdata["metrics"].get("val_loss", [])
        zsrtas = mdata["metrics"].get("ZSRTA", [])

        # derive best/final values as specified
        best_train_acc = max(train_accs) if train_accs else float("nan")
        best_val_acc = max(val_accs) if val_accs else float("nan")
        best_val_loss = min(val_losses) if val_losses else float("nan")
        final_zsrta = zsrtas[-1] if zsrtas else float("nan")

        # output with explicit, descriptive labels --------------------
        print(f"  Hyper-parameter setting: {run_name}")
        print(f"    Train accuracy: {best_train_acc:.4f}")
        print(f"    Validation accuracy: {best_val_acc:.4f}")
        print(f"    Validation loss: {best_val_loss:.4f}")
        print(f"    Zero-shot rule transfer accuracy: {final_zsrta:.4f}")
