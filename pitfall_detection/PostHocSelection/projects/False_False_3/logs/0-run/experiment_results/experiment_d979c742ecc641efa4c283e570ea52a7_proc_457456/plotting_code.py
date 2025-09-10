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
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper ----------
def get_metric_list(exp_key, metric_name):
    vals = experiment_data[exp_key]["SPR_BENCH"]["metrics"]["val"]
    return [m.get(metric_name) if isinstance(m, dict) else None for m in vals]


# ---------- figure 1: loss curves ----------
try:
    plt.figure()
    for k in experiment_data:
        tr = experiment_data[k]["SPR_BENCH"]["losses"]["train"]
        va = experiment_data[k]["SPR_BENCH"]["losses"]["val"]
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, "--", label=f"{k} train")
        plt.plot(epochs, va, "-", label=f"{k} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss\n(dropout comparison)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- figure 2/3/4: metric curves ----------
for metric in ["SWA", "CWA", "HWA"]:
    try:
        plt.figure()
        for k in experiment_data:
            vals = get_metric_list(k, metric)
            epochs = range(1, len(vals) + 1)
            plt.plot(epochs, vals, label=k)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"SPR_BENCH: Validation {metric} vs Epoch\n(dropout comparison)")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_{metric}_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating {metric} curves: {e}")
        plt.close()

# ---------- figure 5: final HWA bar chart ----------
try:
    plt.figure()
    names, final_hwa = [], []
    for k in experiment_data:
        hwa_vals = get_metric_list(k, "HWA")
        if hwa_vals and hwa_vals[-1] is not None:
            names.append(k)
            final_hwa.append(hwa_vals[-1])
    plt.bar(names, final_hwa, color="skyblue")
    plt.ylabel("Final Epoch HWA")
    plt.title("SPR_BENCH: Final Validation HWA per Dropout\n(bar chart)")
    plt.xticks(rotation=45, ha="right")
    fname = os.path.join(working_dir, "SPR_BENCH_final_HWA_bar.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar chart: {e}")
    plt.close()

print("Generated figures:", [f for f in os.listdir(working_dir) if f.endswith(".png")])
