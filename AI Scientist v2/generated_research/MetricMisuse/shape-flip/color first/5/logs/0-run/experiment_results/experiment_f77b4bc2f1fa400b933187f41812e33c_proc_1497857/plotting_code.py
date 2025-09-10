import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------- #
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp_dict = experiment_data["MultiSyntheticGeneralization"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp_dict = {}

# Prepare ordered experiment names for reproducible colors/labels
exp_names = list(exp_dict.keys())
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# --------------------------------------------------------------------- #
# 1) Loss curves
try:
    plt.figure()
    for idx, name in enumerate(exp_names):
        train_loss = exp_dict[name]["losses"]["train"]
        val_loss = exp_dict[name]["losses"]["val"]
        epochs = np.arange(1, len(train_loss) + 1)
        plt.plot(
            epochs,
            train_loss,
            color=colors[idx % len(colors)],
            label=f"{name}-train",
            linewidth=1.5,
        )
        plt.plot(
            epochs,
            val_loss,
            color=colors[idx % len(colors)],
            label=f"{name}-val",
            linestyle="--",
            linewidth=1.5,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("MultiSyntheticGeneralization: Loss Curves (Train vs Validation)")
    plt.legend(fontsize="small")
    save_path = os.path.join(
        working_dir, "MultiSyntheticGeneralization_loss_curves.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# --------------------------------------------------------------------- #
# 2) CmpWA curves
try:
    plt.figure()
    for idx, name in enumerate(exp_names):
        train_cmp = exp_dict[name]["CmpWA_train"]
        val_cmp = exp_dict[name]["CmpWA_val"]
        epochs = np.arange(1, len(train_cmp) + 1)
        plt.plot(
            epochs,
            train_cmp,
            color=colors[idx % len(colors)],
            label=f"{name}-train",
            linewidth=1.5,
        )
        plt.plot(
            epochs,
            val_cmp,
            color=colors[idx % len(colors)],
            label=f"{name}-val",
            linestyle="--",
            linewidth=1.5,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Composite Weighted Accuracy")
    plt.title("MultiSyntheticGeneralization: CmpWA Curves (Train vs Validation)")
    plt.legend(fontsize="small")
    save_path = os.path.join(
        working_dir, "MultiSyntheticGeneralization_CmpWA_curves.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating CmpWA curves: {e}")
    plt.close()

# --------------------------------------------------------------------- #
# 3) Test weighted-accuracy bar chart
try:
    metrics = ["CWA", "SWA", "CmpWA"]
    x = np.arange(len(metrics))
    width = 0.35
    plt.figure()
    for idx, name in enumerate(exp_names):
        vals = [exp_dict[name]["test_metrics"][m] for m in metrics]
        plt.bar(
            x + idx * width,
            vals,
            width=width,
            color=colors[idx % len(colors)],
            label=name,
        )
    plt.xticks(x + width * (len(exp_names) - 1) / 2, metrics)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("MultiSyntheticGeneralization: Test Weighted Accuracies")
    plt.legend(fontsize="small")
    save_path = os.path.join(
        working_dir, "MultiSyntheticGeneralization_test_metrics.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating test metric bars: {e}")
    plt.close()
