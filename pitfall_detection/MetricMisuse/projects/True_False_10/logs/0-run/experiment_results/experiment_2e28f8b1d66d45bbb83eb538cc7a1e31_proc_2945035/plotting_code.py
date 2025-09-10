import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data_dict = experiment_data["learning_rate"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data_dict = {}


# helper
def _sorted_lrs(d):
    return sorted(d.keys(), key=lambda x: float(x.replace("e", "E")))


lrs = _sorted_lrs(data_dict)

# ---------- fig 1: loss curves ----------
try:
    plt.figure()
    for lr in lrs:
        tr = data_dict[lr]["losses"]["train"]
        val = data_dict[lr]["losses"]["val"]
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"train lr={lr}")
        plt.plot(epochs, val, "--", label=f"val lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# collect test metrics once for the three bar plots
test_crwa, test_swa, test_cwa = [], [], []
for lr in lrs:
    metr = data_dict[lr]["metrics"]["test"]
    test_crwa.append(metr["CRWA"])
    test_swa.append(metr["SWA"])
    test_cwa.append(metr["CWA"])


def bar_plot(values, metric_name, fname_suffix):
    try:
        plt.figure()
        plt.bar(range(len(lrs)), values, tick_label=lrs)
        plt.ylabel(metric_name)
        plt.xlabel("Learning Rate")
        plt.title(f"SPR_BENCH: Test {metric_name} vs Learning Rate")
        fname = os.path.join(working_dir, f"SPR_BENCH_{fname_suffix}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating {metric_name} plot: {e}")
        plt.close()


# ---------- figs 2-4 ----------
bar_plot(test_crwa, "CRWA", "CRWA_vs_lr")
bar_plot(test_swa, "SWA", "SWA_vs_lr")
bar_plot(test_cwa, "CWA", "CWA_vs_lr")

# ---------- print summary ----------
print("\n=== Test Metrics Summary ===")
for lr, c1, s1, c2 in zip(lrs, test_crwa, test_swa, test_cwa):
    print(f"lr={lr:>5}: CRWA={c1:.4f} | SWA={s1:.4f} | CWA={c2:.4f}")
