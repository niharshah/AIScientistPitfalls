import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    layer_keys = sorted(
        experiment_data["num_lstm_layers"].keys(), key=lambda k: int(k.split("_")[-1])
    )  # e.g. ['layers_1', 'layers_2', ...]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data, layer_keys = None, []


# ---------- helper to fetch series ----------
def series(key, phase, field):
    return [m[field] for m in experiment_data["num_lstm_layers"][key]["metrics"][phase]]


# ---------- 1. train/val loss curves ----------
try:
    if experiment_data:
        plt.figure()
        for k in layer_keys:
            epochs = np.arange(1, len(series(k, "train", "loss")) + 1)
            plt.plot(epochs, series(k, "train", "loss"), label=f"{k}-train")
            plt.plot(
                epochs,
                experiment_data["num_lstm_layers"][k]["losses"]["val"],
                "--",
                label=f"{k}-val",
            )
        plt.title("SPR_BENCH: Train vs. Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- 2. HWA curves ----------
try:
    if experiment_data:
        plt.figure()
        for k in layer_keys:
            epochs = np.arange(1, len(series(k, "val", "hwa")) + 1)
            plt.plot(epochs, series(k, "val", "hwa"), label=f"{k}")
        plt.title("SPR_BENCH: Validation HWA over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_hwa_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ---------- 3. Final HWA bar chart ----------
try:
    if experiment_data:
        final_hwa = [series(k, "val", "hwa")[-1] for k in layer_keys]
        plt.figure()
        plt.bar(layer_keys, final_hwa, color="skyblue")
        plt.title("SPR_BENCH: Final-Epoch HWA by #LSTM Layers")
        plt.ylabel("HWA")
        plt.xlabel("# LSTM Layers")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()

# ---------- 4. SWA vs CWA scatter ----------
try:
    if experiment_data:
        swa_final = [series(k, "val", "swa")[-1] for k in layer_keys]
        cwa_final = [series(k, "val", "cwa")[-1] for k in layer_keys]
        sizes = [200 + 50 * int(k.split("_")[-1]) for k in layer_keys]
        plt.figure()
        plt.scatter(swa_final, cwa_final, s=sizes)
        for x, y, k in zip(swa_final, cwa_final, layer_keys):
            plt.text(x, y, k)
        plt.title("SPR_BENCH: Final SWA vs. CWA (size ∝ layers)")
        plt.xlabel("Shape Weighted Accuracy")
        plt.ylabel("Color Weighted Accuracy")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa_vs_cwa_scatter.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA vs CWA scatter: {e}")
    plt.close()
