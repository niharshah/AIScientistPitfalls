import matplotlib.pyplot as plt
import numpy as np
import os

# working directory setup --------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data -----------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data["emb_dim_tuning"]["SPR_BENCH"]["runs"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = []


# helper for epoch indices -------------------------------------------------
def epochs_of(run):
    return list(range(1, len(run["losses"]["train"]) + 1))


# per-run plots ------------------------------------------------------------
plot_idx = 0
for run in runs:
    emb_dim = run["emb_dim"]
    ep = epochs_of(run)
    # 1) loss curves -------------------------------------------------------
    try:
        plt.figure()
        plt.plot(ep, run["losses"]["train"], label="train")
        plt.plot(ep, run["losses"]["val"], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"SPR_BENCH Loss vs Epoch (emb_dim={emb_dim})")
        plt.legend()
        fname = f"SPR_BENCH_emb{emb_dim}_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        plot_idx += 1
    except Exception as e:
        print(f"Error creating loss plot for emb_dim={emb_dim}: {e}")
        plt.close()

    if plot_idx >= 5:  # safety guard – though we expect 4 plots here
        break

# summary bar chart (5th figure) -------------------------------------------
try:
    test_accs = [r["test"]["acc"] for r in runs]
    emb_dims = [r["emb_dim"] for r in runs]
    plt.figure()
    plt.bar(range(len(emb_dims)), test_accs, tick_label=emb_dims)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Test Accuracy")
    plt.title("SPR_BENCH Final Test Accuracy by Embedding Dimension")
    fname = "SPR_BENCH_test_accuracy_by_emb_dim.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating summary accuracy bar chart: {e}")
    plt.close()

# console summary ----------------------------------------------------------
print("\n=== Validation & Test Accuracy Summary ===")
for r in runs:
    best_val_acc = max(m["acc"] for m in r["metrics"]["val"])
    print(
        f"emb_dim={r['emb_dim']:>3}: best_val={best_val_acc:.3f} "
        f"test={r['test']['acc']:.3f}"
    )
