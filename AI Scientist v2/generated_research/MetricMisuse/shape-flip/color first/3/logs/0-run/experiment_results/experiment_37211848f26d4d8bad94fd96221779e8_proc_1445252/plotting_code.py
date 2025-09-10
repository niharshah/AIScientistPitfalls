import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    section = experiment_data.get("embed_dim", {}).get("SPR_BENCH", {})
    if not section:
        print("No SPR_BENCH data found.")
    else:
        # -------- pick best embed_dim by last dev BWA ---------
        best_dim, best_bwa = None, -1
        for k, v in section.items():
            bwa_curve = v["metrics"]["val"]
            if bwa_curve and bwa_curve[-1] > best_bwa:
                best_bwa, best_dim = bwa_curve[-1], k
        # ----------------- Plot 1: loss curves ----------------
        try:
            logs = section[best_dim]
            epochs = range(1, len(logs["losses"]["train"]) + 1)
            plt.figure()
            plt.plot(epochs, logs["losses"]["train"], label="Train Loss")
            plt.plot(epochs, logs["losses"]["val"], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f'SPR_BENCH (embed_dim={best_dim.split("_")[-1]})\n'
                "Training vs Validation Loss"
            )
            plt.legend()
            fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print(f"Plot saved: {fname}")
        except Exception as e:
            print(f"Error creating loss curve plot: {e}")
            plt.close()
        # --------------- Plot 2: BWA curves -------------------
        try:
            plt.figure()
            plt.plot(epochs, logs["metrics"]["train"], label="Train BWA")
            plt.plot(epochs, logs["metrics"]["val"], label="Val BWA")
            plt.xlabel("Epoch")
            plt.ylabel("Balanced Weighted Accuracy")
            plt.title(
                f'SPR_BENCH (embed_dim={best_dim.split("_")[-1]})\n'
                "Training vs Validation BWA"
            )
            plt.legend()
            fname = os.path.join(working_dir, "spr_bench_bwa_curves.png")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print(f"Plot saved: {fname}")
        except Exception as e:
            print(f"Error creating BWA curve plot: {e}")
            plt.close()
        # ------- Plot 3: final dev BWA for each dim -----------
        try:
            dims, final_bwas = [], []
            for k, v in section.items():
                dims.append(int(k.split("_")[-1]))
                final_bwas.append(v["metrics"]["val"][-1] if v["metrics"]["val"] else 0)
            order = np.argsort(dims)
            dims = np.array(dims)[order]
            final_bwas = np.array(final_bwas)[order]
            plt.figure()
            plt.bar(range(len(dims)), final_bwas, tick_label=dims)
            plt.xlabel("embed_dim")
            plt.ylabel("Final Dev BWA")
            plt.title("SPR_BENCH\nFinal Dev BWA for each embed_dim")
            fname = os.path.join(working_dir, "spr_bench_embed_dim_comparison.png")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print(f"Plot saved: {fname}")
        except Exception as e:
            print(f"Error creating embed_dim comparison plot: {e}")
            plt.close()
