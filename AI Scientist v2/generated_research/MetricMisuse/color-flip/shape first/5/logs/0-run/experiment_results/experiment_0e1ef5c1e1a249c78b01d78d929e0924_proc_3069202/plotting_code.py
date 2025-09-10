import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

bench_key = ("encoder_embedding_dim", "SPR_BENCH")
dims = sorted(experiment_data.get(bench_key[0], {}).get(bench_key[1], {}).keys())

# Gather data
metrics, losses, final_scores = {}, {}, {}
for d in dims:
    rec = experiment_data[bench_key[0]][bench_key[1]][d]
    metrics[d] = rec["metrics"]["train"]  # list of HSCA per epoch
    losses[d] = rec["losses"]["train"]  # list of loss per epoch
    final_scores[d] = rec["metrics"]["val"][-1] if rec["metrics"]["val"] else np.nan

# ------------------------------------------------------------------
# 1) Training HSCA curves
try:
    plt.figure()
    for d in dims:
        plt.plot(range(1, len(metrics[d]) + 1), metrics[d], label=f"dim={d}")
    plt.xlabel("Epoch")
    plt.ylabel("HSCA")
    plt.title("SPR_BENCH: Training HSCA vs Epoch")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_HSCA_vs_epoch.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HSCA plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Training loss curves
try:
    plt.figure()
    for d in dims:
        plt.plot(range(1, len(losses[d]) + 1), losses[d], label=f"dim={d}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training Loss vs Epoch")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_train_loss_vs_epoch.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Final test HSCA bar chart
try:
    plt.figure()
    xs = np.arange(len(dims))
    ys = [final_scores[d] for d in dims]
    plt.bar(xs, ys, tick_label=dims)
    plt.ylabel("HSCA")
    plt.title("SPR_BENCH: Final Test HSCA per Embedding Dim")
    fname = os.path.join(working_dir, "SPR_BENCH_final_test_HSCA.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HSCA bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------
# Console output of final scores
for d in dims:
    print(f"Embedding dim {d}: Final Test HSCA = {final_scores[d]:.4f}")
