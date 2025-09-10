import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
dropouts = sorted(
    [k for k in experiment_data.keys() if k.startswith("dropout_")],
    key=lambda x: float(x.split("_")[1]),
)

# ----------------- individual loss curves -----------------
for tag in dropouts[:5]:  # there are only 4 tags, but keep guard
    try:
        dset = experiment_data[tag][dataset]
        train_l = dset["losses"]["train"]
        val_l = dset["losses"]["val"]
        plt.figure()
        plt.plot(range(1, len(train_l) + 1), train_l, label="train")
        plt.plot(range(1, len(val_l) + 1), val_l, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dataset} Loss Curves – {tag}")
        plt.legend()
        fname = os.path.join(working_dir, f"{dataset}_loss_curve_{tag}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {tag}: {e}")
        plt.close()

# ----------------- summary bar chart: complexity-weighted accuracy -----------------
try:
    acc_vals = [experiment_data[tag][dataset]["comp_weighted_acc"] for tag in dropouts]
    plt.figure()
    plt.bar(dropouts, acc_vals, color="skyblue")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.ylim(0, 1)
    plt.title(f"{dataset} – Final Complexity-Weighted Accuracy (higher is better)")
    for i, v in enumerate(acc_vals):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    fname = os.path.join(working_dir, f"{dataset}_comp_weighted_accuracy.png")
    plt.savefig(fname)
    plt.close()
    print(
        "Complexity-Weighted Accuracies:",
        dict(zip(dropouts, [round(a, 4) for a in acc_vals])),
    )
except Exception as e:
    print(f"Error creating accuracy bar chart: {e}")
    plt.close()
