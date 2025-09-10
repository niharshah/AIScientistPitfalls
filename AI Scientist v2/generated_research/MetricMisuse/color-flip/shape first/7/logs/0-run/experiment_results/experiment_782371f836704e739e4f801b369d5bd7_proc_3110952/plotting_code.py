import matplotlib.pyplot as plt
import numpy as np
import os

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

# ---------- plot ----------
final_hwas = {}
root = experiment_data.get("static_embedding", {}).get("SPR_BENCH", {})
for hs, run in root.items():
    try:
        epochs_tr = [e for e, _ in run["losses"]["train"]]
        tr_loss = [v for _, v in run["losses"]["train"]]
        val_loss = [v for _, v in run["losses"]["val"]]
        hwa = [v[3] for v in run["metrics"]["val"]]
        final_hwas[hs] = hwa[-1] if hwa else 0.0

        fig, ax1 = plt.subplots()
        ax1.plot(epochs_tr, tr_loss, label="Train Loss", color="tab:blue")
        ax1.plot(epochs_tr, val_loss, label="Val Loss", color="tab:orange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("CrossEntropy Loss")
        ax2 = ax1.twinx()
        ax2.plot(epochs_tr, hwa, label="Val HWA", color="tab:green")
        ax2.set_ylabel("HWA")

        lines, labels = [], []
        for ax in (ax1, ax2):
            h, l = ax.get_legend_handles_labels()
            lines += h
            labels += l
        ax1.legend(lines, labels, loc="upper right")

        title = f"SPR_BENCH StaticEmb Hidden={hs}: Loss & HWA Curves"
        plt.title(title)
        fname = f"SPR_BENCH_static_embedding_hs{hs}_training_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating plot for hidden={hs}: {e}")
    finally:
        plt.close()

# ---- aggregate final HWA comparison (5th plot) ----
try:
    if final_hwas:
        plt.figure()
        hs_sorted = sorted(final_hwas)
        hwa_vals = [final_hwas[h] for h in hs_sorted]
        plt.bar([str(h) for h in hs_sorted], hwa_vals, color="tab:purple")
        plt.xlabel("Hidden Size")
        plt.ylabel("Final HWA")
        plt.title("SPR_BENCH StaticEmb: Final HWA vs Hidden Size")
        fname = "SPR_BENCH_static_embedding_final_HWA_comparison.png"
        plt.savefig(os.path.join(working_dir, fname))
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating final HWA comparison plot: {e}")
finally:
    plt.close()
