import os
import numpy as np

# ---------------------------------------------------------------------
# locate and load the results dictionary
working_dir = os.path.join(os.getcwd(), "working")
exp_file = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(exp_file, allow_pickle=True).item()


# ---------------------------------------------------------------------
def show_metrics(exp_dict):
    """
    Print the final / best metric values stored in the experiment_data structure.
    """
    for dataset_name, info in exp_dict.items():
        print(dataset_name)  # dataset header

        # ---- final losses ------------------------------------------------
        final_train_loss = info["losses"]["train"][-1]
        final_val_loss = info["losses"]["val"][-1]
        print(f"training loss: {final_train_loss:.6f}")
        print(f"validation loss: {final_val_loss:.6f}")

        # ---- final per-split metrics ------------------------------------
        for metric_name in ["CWA", "SWA", "CmpWA"]:
            final_train_metric = info["metrics"]["train"][metric_name][-1]
            final_val_metric = info["metrics"]["val"][metric_name][-1]
            print(f"training {metric_name}: {final_train_metric:.6f}")
            print(f"validation {metric_name}: {final_val_metric:.6f}")

        # ---- test set metrics -------------------------------------------
        if "test" in info and info["test"]:
            test_block = info["test"]
            print(f"test loss: {test_block['loss']:.6f}")
            for metric_name in ["CWA", "SWA", "CmpWA"]:
                if metric_name in test_block:
                    print(f"test {metric_name}: {test_block[metric_name]:.6f}")


# execute immediately
show_metrics(experiment_data)
