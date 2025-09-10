import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["gru_num_layers"]["SPR_BENCH"]
    params = exp["param_values"]
    train_losses = exp["losses"]["train"]  # list[list[float]]
    val_losses = exp["losses"]["val"]
    val_hwas = exp["metrics"]["val"]  # list[list[float]]
    test_hwas = exp["metrics"]["test"]  # list[float]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

# --------------- per-param loss curves -----------------
for i, p in enumerate(params):
    try:
        epochs = range(1, len(train_losses[i]) + 1)
        plt.figure(figsize=(4, 3))
        plt.plot(epochs, train_losses[i], label="Train Loss")
        plt.plot(epochs, val_losses[i], label="Val Loss")
        plt.title(f"SPR_BENCH: Train vs Val Loss\nGRU num_layers={p}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"SPR_BENCH_loss_curves_layers_{p}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for layers={p}: {e}")
        plt.close()

# --------------- validation HWA curves -----------------
for i, p in enumerate(params):
    try:
        epochs = range(1, len(val_hwas[i]) + 1)
        plt.figure(figsize=(4, 3))
        plt.plot(epochs, val_hwas[i], marker="o")
        plt.title(f"SPR_BENCH: Validation HWA\nGRU num_layers={p}")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.ylim(0, 1)
        fname = f"SPR_BENCH_valHWA_layers_{p}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot for layers={p}: {e}")
        plt.close()

# --------------- bar chart of final test HWA -----------------
try:
    plt.figure(figsize=(4, 3))
    plt.bar([str(p) for p in params], test_hwas, color="skyblue")
    plt.title("SPR_BENCH: Test HWA vs GRU num_layers")
    plt.xlabel("GRU num_layers")
    plt.ylabel("Test HWA")
    plt.ylim(0, 1)
    for idx, h in enumerate(test_hwas):
        plt.text(idx, h + 0.01, f"{h:.2f}", ha="center")
    fname = "SPR_BENCH_testHWA_bar.png"
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    print("Best Test HWA:", max(test_hwas))
except Exception as e:
    print(f"Error creating test HWA bar chart: {e}")
    plt.close()
