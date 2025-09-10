import os
import numpy as np

# ------------------------------------------------------------------
# locate and load experiment data
# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
file_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(file_path, allow_pickle=True).item()


# ------------------------------------------------------------------
# helper: pick best or final value + generate pretty metric name
# ------------------------------------------------------------------
def select_value(metric_key, values):
    """
    Return the value to report for a metric list.
    Accuracy-like metrics -> highest, loss-like -> lowest.
    For single-element lists just return that element.
    """
    if not values:  # empty safety-check
        return None
    if len(values) == 1:  # only one entry
        return values[0]
    if "loss" in metric_key.lower():  # minimise losses
        return min(values)
    return max(values)  # maximise accuracies, etc.


def pretty_name(metric_key):
    """
    Convert internal metric keys to readable names.
    """
    mapping = {
        "train_acc": "train accuracy",
        "val_acc": "validation accuracy",
        "val_loss": "validation loss",
        "ZSRTA": "zero-shot rule transfer accuracy",
    }
    return mapping.get(metric_key, metric_key.replace("_", " "))


# ------------------------------------------------------------------
# print results
# ------------------------------------------------------------------
for dset_name, dset_dict in experiment_data.items():
    print(f"{dset_name}:")  # dataset header

    metrics = dset_dict.get("metrics", {})
    for key, values in metrics.items():
        best_val = select_value(key, values)
        if best_val is None:
            continue
        print(f"  {pretty_name(key)}: {best_val:.4f}")

    print()  # blank line between datasets
