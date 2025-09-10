import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to fetch arrays safely
def get_lists(drop_dict, key):
    return [drop_dict[d]["SPR_BENCH"][key] for d in drop_dict]


dropout_dict = experiment_data.get("dropout_rate", {})
dropouts = sorted(dropout_dict.keys())
loss_train = [v["SPR_BENCH"]["losses"]["train"] for v in dropout_dict.values()]
loss_val = [v["SPR_BENCH"]["losses"]["val"] for v in dropout_dict.values()]
gcwa_val = [
    [e["GCWA"] for e in v["SPR_BENCH"]["metrics"]["val"]] for v in dropout_dict.values()
]
test_metrics = {
    d: dropout_dict[d]["SPR_BENCH"]["metrics"]["test"] for d in dropout_dict
}

# ---------- plotting ----------
try:
    plt.figure()
    for d, tr, vl in zip(dropouts, loss_train, loss_val):
        epochs = np.arange(1, len(tr) + 1)
        plt.plot(epochs, tr, "--", label=f"train loss d={d}")
        plt.plot(epochs, vl, "-", label=f"val loss d={d}")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

try:
    plt.figure()
    for d, gc in zip(dropouts, gcwa_val):
        epochs = np.arange(1, len(gc) + 1)
        plt.plot(epochs, gc, label=f"d={d}")
    plt.title("SPR_BENCH: Validation GCWA vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("GCWA")
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, "SPR_BENCH_val_GCWA_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating GCWA plot: {e}")
    plt.close()

try:
    metrics = ["CWA", "SWA", "GCWA"]
    x = np.arange(len(dropouts))
    width = 0.25
    plt.figure()
    for i, m in enumerate(metrics):
        vals = [test_metrics[d][m] for d in dropouts]
        plt.bar(x + i * width - width, vals, width, label=m)
    plt.title("SPR_BENCH: Test Metrics per Dropout")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Score")
    plt.xticks(x, [str(d) for d in dropouts])
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_test_metric_bars.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar plot: {e}")
    plt.close()

# ---------- print table ----------
for d in dropouts:
    m = test_metrics[d]
    print(f"dropout={d}: CWA={m['CWA']:.3f}, SWA={m['SWA']:.3f}, GCWA={m['GCWA']:.3f}")
