import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch dict safely
def get_spr_records(exp_data):
    return exp_data.get("embedding_dim", {}).get("SPR_BENCH", {})


spr_records = get_spr_records(experiment_data)
dims = sorted(spr_records.keys(), key=lambda x: int(x))  # ensure numeric order

# ---------- 1) loss curves per dimension ----------
for dim in dims:
    try:
        rec = spr_records[dim]
        epochs = [e for e, _ in rec["losses"]["train"]]
        tr_loss = [l for _, l in rec["losses"]["train"]]
        va_loss = [l for _, l in rec["losses"]["val"]]
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, va_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"SPR_BENCH Loss Curve (emb_dim={dim})")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_loss_curve_dim{dim}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for dim {dim}: {e}")
        plt.close()

# ---------- 2) bar chart of test metrics ----------
try:
    metrics = {"CWA": [], "SWA": [], "EWA": []}
    for dim in dims:
        tm = spr_records[dim]["test_metrics"]
        for k in metrics:
            metrics[k].append(tm[k])
    x = np.arange(len(dims))
    width = 0.25
    plt.figure()
    for i, (k, vals) in enumerate(metrics.items()):
        plt.bar(x + i * width - width, vals, width, label=k)
    plt.xticks(x, dims)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Score")
    plt.title("SPR_BENCH Test Metrics by Embedding Dimension")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar plot: {e}")
    plt.close()

# ---------- 3) dev-set EWA line plot ----------
try:
    dev_ewa = [spr_records[dim]["metrics"]["val"][-1][1]["EWA"] for dim in dims]
    plt.figure()
    plt.plot([int(d) for d in dims], dev_ewa, marker="o")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Dev-set EWA")
    plt.title("SPR_BENCH Dev EWA vs Embedding Dimension")
    fname = os.path.join(working_dir, "SPR_BENCH_dev_EWA_vs_dim.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating dev EWA plot: {e}")
    plt.close()
