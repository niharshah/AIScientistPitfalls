import os
import numpy as np

# -------- locate and load experiment data ---------
working_dir = os.path.join(os.getcwd(), "working")
exp_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(exp_path, allow_pickle=True).item()

# -------- iterate and report metrics --------------
for model_name, datasets in experiment_data.items():
    for dataset_name, rec in datasets.items():
        print(dataset_name)  # dataset header

        # Training loss (final value)
        train_losses = rec.get("losses", {}).get("train", [])
        if train_losses:
            print(f"final training loss: {train_losses[-1]:.4f}")

        # Validation loss (best value)
        val_losses = rec.get("losses", {}).get("val", [])
        if val_losses:
            best_val_loss = min(val_losses)
            print(f"best validation loss: {best_val_loss:.4f}")

        # Validation SWA (best value)
        val_swa_scores = rec.get("SWA", {}).get("val", [])
        if val_swa_scores:
            best_val_swa = max(val_swa_scores)
            print(f"best validation shape-weighted accuracy: {best_val_swa:.3f}")

        # Test metrics (single recorded values)
        test_metrics = rec.get("metrics", {}).get("test", {})
        if "loss" in test_metrics:
            print(f"test loss: {test_metrics['loss']:.4f}")
        if "SWA" in test_metrics:
            print(f"test shape-weighted accuracy: {test_metrics['SWA']:.3f}")
