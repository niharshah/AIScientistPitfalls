import matplotlib.pyplot as plt
import numpy as np
import os

# set up dirs
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment results ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

hidden_dict = experiment_data.get("hidden_size", {})

# store final test metrics for summary plot / printing
summary = []

# ---------- per-hidden_size curves ----------
for hsz, data in hidden_dict.items():
    run = data.get("SPR_BENCH", {})
    losses = run.get("losses", {})
    metrics = run.get("metrics", {})
    train_loss = losses.get("train", [])  # list of (epoch, value)
    val_loss = losses.get("val", [])
    val_dwh = [(e, dwh) for e, _, _, dwh in metrics.get("val", [])]

    try:
        epochs_loss = [e for e, _ in train_loss]
        tl_vals = [v for _, v in train_loss]
        vl_vals = [v for _, v in val_loss]
        dwh_vals = [d for _, d in val_dwh]

        fig, ax1 = plt.subplots()
        ax1.plot(epochs_loss, tl_vals, "b-", label="Train Loss")
        ax1.plot(epochs_loss, vl_vals, "r-", label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross-Entropy Loss")
        ax1.tick_params(axis="y")
        ax2 = ax1.twinx()
        ax2.plot(epochs_loss, dwh_vals, "g--", label="Val DWHS")
        ax2.set_ylabel("DWHS")
        ax2.tick_params(axis="y")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        plt.title(f"SPR_BENCH | hidden={hsz}\nLeft: Train/Val Loss, Right: Val DWHS")
        fname = os.path.join(working_dir, f"SPR_BENCH_hidden{hsz}_loss_metric.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot for hidden={hsz}: {e}")
        plt.close()

    # collect final test metrics
    test_cwa, test_swa, test_dwh = run.get("metrics", {}).get(
        "test", (None, None, None)
    )
    summary.append((hsz, test_cwa, test_swa, test_dwh))

# ---------- summary plot ----------
try:
    if summary:
        summary = sorted(summary, key=lambda x: x[0])  # sort by hidden_size
        hszs = [s[0] for s in summary]
        dwhs = [s[3] for s in summary]
        plt.figure()
        plt.plot(hszs, dwhs, marker="o")
        plt.xlabel("Hidden Size")
        plt.ylabel("Test DWHS")
        plt.title("SPR_BENCH | Final Test DWHS vs Hidden Size")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_DWHS_vs_hidden.png"))
        plt.close()
except Exception as e:
    print(f"Error creating summary plot: {e}")
    plt.close()

# ---------- print results ----------
if summary:
    print("\nFinal Test Metrics per hidden_size (CWA, SWA, DWHS):")
    for h, c, s, d in summary:
        print(f"  hidden={h:3d} | CWA={c:.3f} SWA={s:.3f} DWHS={d:.3f}")
    best = max(summary, key=lambda x: x[3] if x[3] is not None else -1)
    print(f"\nBest hidden_size by DWHS: {best[0]} (DWHS={best[3]:.3f})")
