import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    layer_keys = sorted(
        experiment_data["num_gru_layers"].keys(), key=lambda x: int(x.split("_")[-1])
    )  # ['layers_1', ...]
    # ---------- plot 1: loss curves ----------
    try:
        plt.figure(figsize=(6, 4))
        for k in layer_keys:
            tloss = experiment_data["num_gru_layers"][k]["losses"]["train"]
            vloss = experiment_data["num_gru_layers"][k]["losses"]["val"]
            epochs = range(1, len(tloss) + 1)
            plt.plot(epochs, tloss, marker="o", label=f"Train {k}")
            plt.plot(epochs, vloss, marker="x", linestyle="--", label=f"Val {k}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- plot 2: test metrics ----------
    try:
        metrics = ["acc", "swa", "cwa", "nrgs"]
        bar_width = 0.2
        x_base = np.arange(len(metrics))
        plt.figure(figsize=(7, 4))
        for idx, k in enumerate(layer_keys):
            res = experiment_data["num_gru_layers"][k]["metrics"]["test"]
            vals = [res[m] for m in metrics]
            plt.bar(x_base + idx * bar_width, vals, width=bar_width, label=k)
        plt.xticks(x_base + bar_width, [m.upper() for m in metrics])
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Test Metrics by GRU Depth")
        plt.legend(title="Model")
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_test_metrics.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()
