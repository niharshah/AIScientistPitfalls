import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    records = experiment_data.get("num_gnn_layers", {})
    depths = sorted(
        int(k.split("_")[-1])
        for k in records.keys()
        if k.startswith("SPR_BENCH_layers_")
    )

    # Helper to fetch series
    def get_series(metric_key, split, depth):
        rec = records[f"SPR_BENCH_layers_{depth}"]
        return rec[metric_key][split]

    # 1) Train/Val Loss Curves
    try:
        plt.figure()
        for depth in depths:
            tr = get_series("losses", "train", depth)
            val = get_series("losses", "val", depth)
            epochs = range(1, len(tr) + 1)
            plt.plot(epochs, tr, label=f"Train depth={depth}")
            plt.plot(epochs, val, "--", label=f"Val depth={depth}")
        plt.title("SPR_BENCH: Train vs Val Loss across GNN depths")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2) Train/Val Accuracy Curves
    try:
        plt.figure()
        for depth in depths:
            tr = [m["acc"] for m in get_series("metrics", "train", depth)]
            val = [m["acc"] for m in get_series("metrics", "val", depth)]
            epochs = range(1, len(tr) + 1)
            plt.plot(epochs, tr, label=f"Train depth={depth}")
            plt.plot(epochs, val, "--", label=f"Val depth={depth}")
        plt.title("SPR_BENCH: Train vs Val Accuracy across GNN depths")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 3) Train/Val Complexity-Weighted Accuracy (CoWA) Curves
    try:
        plt.figure()
        for depth in depths:
            tr = [m["cowa"] for m in get_series("metrics", "train", depth)]
            val = [m["cowa"] for m in get_series("metrics", "val", depth)]
            epochs = range(1, len(tr) + 1)
            plt.plot(epochs, tr, label=f"Train depth={depth}")
            plt.plot(epochs, val, "--", label=f"Val depth={depth}")
        plt.title("SPR_BENCH: Train vs Val CoWA across GNN depths")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_CoWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CoWA plot: {e}")
        plt.close()

    # 4) Test-set summary bar chart
    try:
        plt.figure(figsize=(6, 4))
        accs = [records[f"SPR_BENCH_layers_{d}"]["test"]["acc"] for d in depths]
        cows = [records[f"SPR_BENCH_layers_{d}"]["test"]["cowa"] for d in depths]
        x = np.arange(len(depths))
        w = 0.35
        plt.bar(x - w / 2, accs, width=w, label="Accuracy")
        plt.bar(x + w / 2, cows, width=w, label="CoWA")
        plt.xticks(x, [str(d) for d in depths])
        plt.xlabel("Number of GNN Layers")
        plt.ylabel("Test Metric Value")
        plt.title("SPR_BENCH: Test Accuracy and CoWA vs Depth")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metric plot: {e}")
        plt.close()
