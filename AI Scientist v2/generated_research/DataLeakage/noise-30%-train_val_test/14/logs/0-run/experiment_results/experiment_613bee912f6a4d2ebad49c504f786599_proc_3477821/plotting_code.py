import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to extract series
def get_series(exp_dict, key):
    # returns epochs, train_vals, val_vals for given key path ("losses" or "metrics")
    tr_list = exp_dict[key]["train"]
    val_list = exp_dict[key]["val"]
    epochs = [d["epoch"] for d in tr_list]
    tr_vals = [
        d[list(d.keys() - {"epoch"})[0]] for d in tr_list
    ]  # first non-epoch field
    val_vals = [d[list(d.keys() - {"epoch"})[0]] for d in val_list]
    return epochs, tr_vals, val_vals


dataset_name = "SPR_BENCH"
models = experiment_data.get(dataset_name, {})

# ------------- individual model plots -------------
for model_name, exp in models.items():
    try:
        epochs, tr_loss, val_loss = get_series(exp, "losses")
        _, tr_f1, val_f1 = get_series(exp, "metrics")
        plt.figure(figsize=(10, 4))

        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()

        # Macro F1 subplot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, tr_f1, label="Train")
        plt.plot(epochs, val_f1, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("Macro F1")
        plt.legend()

        plt.suptitle(f"{dataset_name} – {model_name}: Loss & Macro F1")
        fname = f"{dataset_name}_{model_name}_loss_f1_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {model_name}: {e}")
        plt.close()

# ------------- aggregated comparison plot -------------
try:
    plt.figure()
    for model_name, exp in models.items():
        epochs, _, val_f1 = get_series(exp, "metrics")
        plt.plot(epochs, val_f1, label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Macro F1")
    plt.title(f"{dataset_name}: Validation Macro F1 Comparison")
    plt.legend()
    fname = f"{dataset_name}_val_macroF1_comparison.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating comparison plot: {e}")
    plt.close()
