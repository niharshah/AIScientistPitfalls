import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data ---------------- #
exp_path_candidates = [
    os.path.join(os.getcwd(), "experiment_data.npy"),
    os.path.join(working_dir, "experiment_data.npy"),
]
experiment_data = None
for p in exp_path_candidates:
    try:
        experiment_data = np.load(p, allow_pickle=True).item()
        break
    except Exception:
        pass

if experiment_data is None:
    print("Could not locate experiment_data.npy")
    exit()

# ---------------- parse metrics ---------------- #
dataset = "SPR_BENCH"
bs_entries = experiment_data["batch_size"].get(dataset, {})
batch_sizes = sorted(bs_entries.keys(), key=int)

loss_curves = {}
cwa_curves = {}
test_metrics = {}

for bs in batch_sizes:
    entry = bs_entries[bs]
    loss_curves[int(bs)] = (entry["losses"]["train"], entry["losses"]["val"])
    cwa = [m["CWA"] for m in entry["metrics"]["val"]]
    cwa_curves[int(bs)] = cwa
    test_metrics[int(bs)] = entry["metrics"]["test"]


# ---------------- plotting utilities ---------------- #
def save_fig(fig, fname):
    fig.savefig(os.path.join(working_dir, fname), bbox_inches="tight")
    plt.close(fig)


# ---------------- 1. loss curves ---------------- #
try:
    fig = plt.figure()
    for bs in batch_sizes:
        t, v = loss_curves[int(bs)]
        epochs = range(1, len(t) + 1)
        plt.plot(epochs, t, "--", label=f"{bs}-train")
        plt.plot(epochs, v, "-", label=f"{bs}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Training vs Validation Loss")
    plt.legend()
    save_fig(fig, "SPR_BENCH_loss_curves.png")
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------------- 2. CWA curves ---------------- #
try:
    fig = plt.figure()
    for bs in batch_sizes:
        cwa = cwa_curves[int(bs)]
        epochs = range(1, len(cwa) + 1)
        plt.plot(epochs, cwa, label=f"{bs}")
    plt.xlabel("Epoch")
    plt.ylabel("CWA")
    plt.title("SPR_BENCH Validation Color-Weighted Accuracy (CWA)")
    plt.legend(title="Batch Size")
    save_fig(fig, "SPR_BENCH_CWA_curves.png")
except Exception as e:
    print(f"Error creating CWA curves: {e}")
    plt.close()

# ---------------- 3. test metric bars ---------------- #
try:
    fig, ax = plt.subplots()
    ind = np.arange(len(batch_sizes))
    width = 0.25
    cwa_vals = [test_metrics[int(bs)]["CWA"] for bs in batch_sizes]
    swa_vals = [test_metrics[int(bs)]["SWA"] for bs in batch_sizes]
    gcwa_vals = [test_metrics[int(bs)]["GCWA"] for bs in batch_sizes]
    ax.bar(ind - width, cwa_vals, width, label="CWA")
    ax.bar(ind, swa_vals, width, label="SWA")
    ax.bar(ind + width, gcwa_vals, width, label="GCWA")
    ax.set_xticks(ind)
    ax.set_xticklabels(batch_sizes)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Score")
    ax.set_title("SPR_BENCH Test Metrics by Batch Size")
    ax.legend()
    save_fig(fig, "SPR_BENCH_test_metric_bars.png")
except Exception as e:
    print(f"Error creating test metric bar chart: {e}")
    plt.close()

# ---------------- print numerical summary ---------------- #
print("\n=== SPR_BENCH Test Metrics ===")
for bs in batch_sizes:
    tm = test_metrics[int(bs)]
    print(f'BS {bs}: CWA={tm["CWA"]:.3f}, SWA={tm["SWA"]:.3f}, GCWA={tm["GCWA"]:.3f}')
