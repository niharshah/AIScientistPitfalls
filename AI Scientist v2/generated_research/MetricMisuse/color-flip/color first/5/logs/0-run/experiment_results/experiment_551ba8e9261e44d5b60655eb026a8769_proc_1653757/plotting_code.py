import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    bench = experiment_data["glyph_cluster_mlp"]["SPR_BENCH"]
    losses = bench["losses"]
    val_metrics = bench["metrics"]["val"]
    test_metrics = bench["metrics"]["test"]
    cluster_info = bench["cluster_info"]

    # 1. Train vs Val loss
    try:
        plt.figure(figsize=(6, 4))
        epochs = np.arange(1, len(losses["train"]) + 1)
        plt.plot(epochs, losses["train"], label="train")
        plt.plot(epochs, losses["val"], linestyle="--", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH — Loss Curves\n(Left: train solid, Right: val dashed)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2. Validation metric curves
    try:
        plt.figure(figsize=(6, 4))
        met_names = ["acc", "cwa", "swa", "ccwa"]
        for m in met_names:
            vals = [d[m] for d in val_metrics]
            plt.plot(epochs, vals, label=m.upper())
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.title(
            "SPR_BENCH — Validation Metrics over Epochs\n(Left: raw, Right: weighted)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating val metric plot: {e}")
        plt.close()

    # 3. Test metrics bar chart
    try:
        plt.figure(figsize=(6, 4))
        vals = [test_metrics[k] for k in met_names]
        plt.bar(met_names, vals, color="skyblue")
        plt.ylim(0, 1)
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.title("SPR_BENCH — Test Metrics\n(Left: Ground Truth, Right: Model Output)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metric plot: {e}")
        plt.close()

    # 4. Silhouette per cluster
    try:
        plt.figure(figsize=(6, 4))
        cl_ids = sorted(cluster_info["cluster_sil"].keys())
        sil_vals = [cluster_info["cluster_sil"][cl] for cl in cl_ids]
        plt.bar([str(c) for c in cl_ids], sil_vals, color="salmon")
        plt.xlabel("Cluster ID")
        plt.ylabel("Mean Silhouette")
        plt.ylim(0, 1)
        plt.title(
            "SPR_BENCH — Glyph Cluster Quality\n(Left: clusters, Right: silhouette)"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_cluster_silhouette.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating silhouette plot: {e}")
        plt.close()

    # -------- Print test metrics --------
    if test_metrics:
        for k, v in test_metrics.items():
            print(f"{k.upper():5s}: {v:.3f}")
