import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data.get("num_epochs", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}


# helper to fetch per-run arrays
def get_arrays(key1, key2):
    return {run: np.asarray(runs[run][key1][key2]) for run in runs}


epochs_dict = {run: np.asarray(runs[run]["epochs"]) for run in runs}
train_scwa = get_arrays("metrics", "train")
val_scwa = get_arrays("metrics", "val")
train_loss = get_arrays("losses", "train")
val_loss = get_arrays("losses", "val")
test_scwa = {run: runs[run].get("test_SCWA", np.nan) for run in runs}

# 1) SCWA curves
try:
    plt.figure()
    for run in runs:
        plt.plot(epochs_dict[run], train_scwa[run], label=f"{run}-train")
        plt.plot(epochs_dict[run], val_scwa[run], label=f"{run}-val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("SCWA")
    plt.title("SPR_BENCH - Training vs Validation SCWA Curves")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_SCWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SCWA plot: {e}")
    plt.close()

# 2) Loss curves
try:
    plt.figure()
    for run in runs:
        plt.plot(epochs_dict[run], train_loss[run], label=f"{run}-train")
        plt.plot(epochs_dict[run], val_loss[run], label=f"{run}-val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH - Training vs Validation Loss Curves")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) Test SCWA bar chart
try:
    plt.figure()
    names = list(test_scwa.keys())
    scores = [test_scwa[n] for n in names]
    plt.bar(range(len(names)), scores, tick_label=names)
    plt.ylabel("Test SCWA")
    plt.title("SPR_BENCH - Test SCWA by Number of Epochs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_test_SCWA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test SCWA bar plot: {e}")
    plt.close()

# print numerical summary
print("Run Name | Test SCWA")
for run, score in test_scwa.items():
    print(f"{run:<25} {score:.4f}")
