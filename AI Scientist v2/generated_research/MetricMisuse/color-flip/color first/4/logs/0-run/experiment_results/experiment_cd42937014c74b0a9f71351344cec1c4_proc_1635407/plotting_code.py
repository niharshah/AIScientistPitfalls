import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Gather data
lrs = sorted(
    [k for k in experiment_data if k.startswith("lr_")],
    key=lambda x: float(x.split("_")[1]),
)
losses_train, losses_val, accs, cwas, swas, pcwas = {}, {}, {}, {}, {}, {}

for lr in lrs:
    rec = experiment_data[lr]["SPR_BENCH"]
    losses_train[lr] = rec["losses"]["train"]
    losses_val[lr] = rec["losses"]["val"]
    # convert list of dicts to metric lists
    m = rec["metrics"]["val"]
    accs[lr] = [d["acc"] for d in m]
    cwas[lr] = [d["cwa"] for d in m]
    swas[lr] = [d["swa"] for d in m]
    pcwas[lr] = [d["pcwa"] for d in m]

epochs = range(1, len(next(iter(losses_train.values()))) + 1)


def plot_metric(metric_dict, ylabel, filename):
    try:
        plt.figure()
        for lr, vals in metric_dict.items():
            plt.plot(epochs, vals, label=f"lr={lr.split('_')[1]}")
        plt.title(f"SPR_BENCH {ylabel} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        save_path = os.path.join(working_dir, f"SPR_BENCH_{filename}.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating plot {filename}: {e}")
        plt.close()


# 1. Loss curves
plot_metric(
    {k: losses_train[k] for k in lrs} | {f"{k}_val": losses_val[k] for k in lrs},
    "Loss",
    "loss_curves",
)

# 2. Validation Accuracy
plot_metric(accs, "Validation Accuracy", "val_accuracy")

# 3. Color-Weighted Accuracy
plot_metric(cwas, "Color-Weighted Accuracy", "cwa")

# 4. Shape-Weighted Accuracy
plot_metric(swas, "Shape-Weighted Accuracy", "swa")

# 5. PC-Weighted Accuracy
plot_metric(pcwas, "PC-Weighted Accuracy", "pcwa")

# Print final epoch metrics for quick inspection
for lr in lrs:
    print(
        f"{lr}: ACC={accs[lr][-1]:.3f}, CWA={cwas[lr][-1]:.3f}, "
        f"SWA={swas[lr][-1]:.3f}, PCWA={pcwas[lr][-1]:.3f}"
    )
