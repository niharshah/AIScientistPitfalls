import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

depths = sorted(experiment_data.get("layer_depth", {}).keys())


# Helper to extract series
def unpack_losses(depth, split):
    lst = experiment_data["layer_depth"][depth]["SPR_BENCH"]["losses"][split]
    epochs, vals = zip(*lst)
    return np.array(epochs), np.array(vals)


def unpack_metric(depth, idx):  # idx: 0=SWA,1=CWA,2=HWA
    lst = experiment_data["layer_depth"][depth]["SPR_BENCH"]["metrics"]["val"]
    epochs, s, c, h = zip(*lst)
    vals = [s, c, h][idx]
    return np.array(epochs), np.array(vals)


# ---------------- plotting -------------------
# 1) Loss curves
try:
    plt.figure()
    for d in depths:
        ep, tr = unpack_losses(d, "train")
        _, va = unpack_losses(d, "val")
        plt.plot(ep, tr, label=f"depth {d}-train", linestyle="-")
        plt.plot(ep, va, label=f"depth {d}-val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_by_depth.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# 2) HWA over epochs
try:
    plt.figure()
    for d in depths:
        ep, hwa_vals = unpack_metric(d, 2)
        plt.plot(ep, hwa_vals, marker="o", label=f"depth {d}")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("SPR_BENCH: HWA vs Epoch for Different LSTM Depths")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_HWA_curves_by_depth.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve figure: {e}")
    plt.close()

# 3) Final HWA bar plot
try:
    plt.figure()
    final_hwa = []
    for d in depths:
        _, hwa_vals = unpack_metric(d, 2)
        final_hwa.append(hwa_vals[-1])
    plt.bar([str(d) for d in depths], final_hwa, color="skyblue")
    plt.xlabel("LSTM Depth")
    plt.ylabel("Final-Epoch HWA")
    plt.title("SPR_BENCH: Final HWA by LSTM Depth")
    fname = os.path.join(working_dir, "SPR_BENCH_final_HWA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar figure: {e}")
    plt.close()

# --------------- print summary metrics ---------------
for d in depths:
    _, swa_vals = unpack_metric(d, 0)
    _, cwa_vals = unpack_metric(d, 1)
    _, hwa_vals = unpack_metric(d, 2)
    print(
        f"Depth {d}: final SWA={swa_vals[-1]:.4f}, "
        f"CWA={cwa_vals[-1]:.4f}, HWA={hwa_vals[-1]:.4f}"
    )
