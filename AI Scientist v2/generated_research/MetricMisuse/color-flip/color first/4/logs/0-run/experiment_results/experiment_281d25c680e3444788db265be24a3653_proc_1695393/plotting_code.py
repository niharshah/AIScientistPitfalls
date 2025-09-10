import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper to get metric arrays ----------
def get_vals(ablation, metric_name):
    m = experiment_data[ablation]["SPR_BENCH"]["metrics"]["val"]
    return [d[metric_name] for d in m]


# ---------- 1. loss curves ----------
try:
    plt.figure()
    for abl, run in experiment_data.items():
        tr_loss = run["SPR_BENCH"]["losses"]["train"]
        val_loss = run["SPR_BENCH"]["losses"]["val"]
        epochs = list(range(1, len(tr_loss) + 1))
        plt.plot(epochs, tr_loss, "--", label=f"{abl} train")
        plt.plot(epochs, val_loss, "-", label=f"{abl} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ---------- 2. accuracy curves ----------
try:
    plt.figure()
    for abl in experiment_data:
        acc = get_vals(abl, "acc")
        epochs = list(range(1, len(acc) + 1))
        plt.plot(epochs, acc, marker="o", label=abl)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("SPR_BENCH: Validation Accuracy per Epoch")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# ---------- 3. final CWA/SWA/CompWA bar chart ----------
try:
    metrics = ["CWA", "SWA", "CompWA"]
    x = np.arange(len(experiment_data))  # ablations
    width = 0.25
    plt.figure()
    for i, met in enumerate(metrics):
        vals = [get_vals(abl, met)[-1] for abl in experiment_data]
        plt.bar(x + i * width, vals, width, label=met)
    plt.xticks(x + width, list(experiment_data.keys()), rotation=45)
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Final Epoch Weighted Metrics")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_final_weighted_metrics.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating weighted metric bar chart: {e}")
    plt.close()

# ---------- print final metrics ----------
for abl in experiment_data:
    final = experiment_data[abl]["SPR_BENCH"]["metrics"]["val"][-1]
    print(
        f"{abl} | ACC={final['acc']:.3f} | CWA={final['CWA']:.3f} | "
        f"SWA={final['SWA']:.3f} | CompWA={final['CompWA']:.3f}"
    )
