import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch metric array safely
def get_metric(act, metric):
    return np.array(
        experiment_data["activation_function"]["SPR_BENCH"][act]["metrics"].get(
            metric, []
        )
    )


acts = list(experiment_data.get("activation_function", {}).get("SPR_BENCH", {}).keys())
epochs = (
    np.arange(1, len(get_metric(acts[0], "train_acc")) + 1) if acts else np.array([])
)

# 1) Accuracy Curves
try:
    plt.figure()
    for act in acts:
        plt.plot(
            epochs, get_metric(act, "train_acc"), label=f"{act}-train", linestyle="-"
        )
        plt.plot(epochs, get_metric(act, "val_acc"), label=f"{act}-val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy Curves (Train vs Val)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2) URA Curves
try:
    plt.figure()
    for act in acts:
        plt.plot(epochs, get_metric(act, "val_ura"), label=act)
    plt.xlabel("Epoch")
    plt.ylabel("URA")
    plt.title("SPR_BENCH Validation URA Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_ura_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating URA curve plot: {e}")
    plt.close()

# 3) Training Loss Curves
try:
    plt.figure()
    for act in acts:
        losses = np.array(
            experiment_data["activation_function"]["SPR_BENCH"][act]["losses"].get(
                "train", []
            )
        )
        plt.plot(epochs, losses, label=act)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Training Loss Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_training_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 4) Test Accuracy Bar Chart
try:
    plt.figure()
    test_accs = [
        experiment_data["activation_function"]["SPR_BENCH"][act]["metrics"].get(
            "test_acc", 0
        )
        for act in acts
    ]
    plt.bar(acts, test_accs)
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Test Accuracy by Activation")
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bars.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar chart: {e}")
    plt.close()

# 5) Test URA Bar Chart
try:
    plt.figure()
    test_uras = [
        experiment_data["activation_function"]["SPR_BENCH"][act]["metrics"].get(
            "test_ura", 0
        )
        for act in acts
    ]
    plt.bar(acts, test_uras, color="orange")
    plt.ylabel("URA")
    plt.title("SPR_BENCH Test URA by Activation")
    fname = os.path.join(working_dir, "SPR_BENCH_test_ura_bars.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating test URA bar chart: {e}")
    plt.close()
