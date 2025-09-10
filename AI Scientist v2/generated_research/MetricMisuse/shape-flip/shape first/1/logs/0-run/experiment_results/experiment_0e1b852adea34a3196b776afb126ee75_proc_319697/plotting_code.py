import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- setup --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to gather hidden dims available
def get_hidden_dims(dct):
    return sorted(int(k.split("_")[-1]) for k in dct.keys())


# -------------------- 1. training / val curves --------------------
try:
    ds_name = "SPR_BENCH"
    ds_runs = experiment_data["multi_dataset_training"][ds_name]
    hds = get_hidden_dims(ds_runs)
    plt.figure()
    for hd in hds:
        hist = ds_runs[f"hidden_{hd}"]["train_curve"]
        tr = hist["metrics"]["train_acc"]
        val = hist["metrics"]["val_acc"]
        epochs = np.arange(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"train_hd{hd}")
        plt.plot(epochs, val, "--", label=f"val_hd{hd}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Training vs Validation Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, f"SPR_BENCH_train_val_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating train/val plot: {e}")
    plt.close()

# -------------------- 2. Test ACC vs hidden dim --------------------
try:
    plt.figure()
    for ds_name in experiment_data["multi_dataset_training"].keys():
        ds_runs = experiment_data["multi_dataset_training"][ds_name]
        hds = get_hidden_dims(ds_runs)
        accs = [ds_runs[f"hidden_{hd}"]["metrics"]["test_acc"] for hd in hds]
        plt.plot(hds, accs, marker="o", label=ds_name)
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Hidden Dimension across Datasets")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "AllDatasets_test_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating ACC plot: {e}")
    plt.close()

# -------------------- 3. SWA and 4. CWA vs hidden dim (combined) ----
for metric, fname_stub in [
    ("SWA", "shape_weighted_accuracy"),
    ("CWA", "color_weighted_accuracy"),
]:
    try:
        plt.figure()
        for ds_name in experiment_data["multi_dataset_training"].keys():
            ds_runs = experiment_data["multi_dataset_training"][ds_name]
            hds = get_hidden_dims(ds_runs)
            vals = [ds_runs[f"hidden_{hd}"]["metrics"][metric] for hd in hds]
            plt.plot(hds, vals, marker="o", label=ds_name)
        plt.xlabel("Hidden Dimension")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Hidden Dimension across Datasets")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"AllDatasets_{fname_stub}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {metric} plot: {e}")
        plt.close()

# -------------------- 5. Print best configuration summary ----------
print("\nBest hidden dimension per dataset (highest test accuracy):")
for ds_name, ds_runs in experiment_data.get("multi_dataset_training", {}).items():
    best_hd, best_acc = None, -1
    for hd in get_hidden_dims(ds_runs):
        acc = ds_runs[f"hidden_{hd}"]["metrics"]["test_acc"]
        if acc > best_acc:
            best_acc, best_hd = acc, hd
    if best_hd is not None:
        met = ds_runs[f"hidden_{best_hd}"]["metrics"]
        print(
            f"{ds_name}: hd={best_hd}, ACC={met['test_acc']:.4f}, "
            f"SWA={met['SWA']:.4f}, CWA={met['CWA']:.4f}, ZSRTA={met['ZSRTA']:.4f}"
        )
