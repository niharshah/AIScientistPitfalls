import os
import numpy as np

# ---------------------------------------------------------------------------
# Locate and load the saved experiment data
working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(file_path, allow_pickle=True).item()

# ---------------------------------------------------------------------------
# Iterate through the stored results and print concise metric summaries
for sweep_name, datasets in experiment_data.items():  # e.g. "num_layers"
    for dataset_name, ds_info in datasets.items():  # e.g. "SPR_BENCH"
        print(f"{dataset_name}")  # dataset header

        # Best hyper-parameter setting discovered during the sweep
        best_layer = ds_info.get("best_layer")
        if best_layer is not None:
            print(f"best number of layers: {best_layer}")

        # Retrieve losses for the best layer (if available)
        if best_layer in ds_info.get("per_layer", {}):
            layer_record = ds_info["per_layer"][best_layer]
            train_losses = layer_record["losses"]["train"]
            val_losses = layer_record["losses"]["val"]

            if train_losses:
                print(f"final training loss: {train_losses[-1]:.4f}")
            if val_losses:
                print(f"final validation loss: {val_losses[-1]:.4f}")

        # Best validation SCWA achieved during hyper-parameter tuning
        best_val_scwa = ds_info.get("best_val_scwa")
        if best_val_scwa is not None:
            print(f"best validation SCWA: {best_val_scwa:.4f}")

        # Test-set SCWA for the model chosen via validation performance
        test_scwa = ds_info.get("test_scwa")
        if test_scwa is not None:
            print(f"test SCWA: {test_scwa:.4f}")
