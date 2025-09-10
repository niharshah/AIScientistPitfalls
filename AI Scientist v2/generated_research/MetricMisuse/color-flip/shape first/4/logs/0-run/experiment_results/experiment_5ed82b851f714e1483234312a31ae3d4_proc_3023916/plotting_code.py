import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
exp_path_try = [os.path.join(working_dir, "experiment_data.npy"), "experiment_data.npy"]
experiment_data = None
for p in exp_path_try:
    if os.path.exists(p):
        experiment_data = np.load(p, allow_pickle=True).item()
        break
if experiment_data is None:
    raise FileNotFoundError("Could not locate experiment_data.npy")

root_key = "num_lstm_layers"
entries = experiment_data[root_key]
layer_tags = sorted(entries.keys())  # e.g. ['SPR_BENCH_layers1',...]

# ---------- convenience collectors ----------
epoch_axis = None
loss_train = {}
loss_val = {}
hwa_val = {}

for tag in layer_tags:
    log = entries[tag]
    loss_train[tag] = log["losses"]["train"]
    loss_val[tag] = log["losses"]["val"]
    hwa_val[tag] = log["metrics"]["val"]
    if epoch_axis is None:
        epoch_axis = np.arange(1, len(loss_train[tag]) + 1)

# -------------------- plot 1: Loss curves --------------------
try:
    plt.figure()
    for tag in layer_tags:
        plt.plot(epoch_axis, loss_train[tag], label=f"{tag}-train")
        plt.plot(epoch_axis, loss_val[tag], "--", label=f"{tag}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss (tuned LSTM layers)")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------------------- plot 2: HWA across epochs --------------
try:
    plt.figure()
    for tag in layer_tags:
        plt.plot(epoch_axis, hwa_val[tag], label=tag)
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH: Validation HWA vs Epoch for different LSTM layers")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_hwa_epochs.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()
except Exception as e:
    print(f"Error creating HWA plot: {e}")
    plt.close()

# -------------------- plot 3: Final HWA bar chart ------------
try:
    plt.figure()
    final_hwa = [hwa_val[tag][-1] for tag in layer_tags]
    plt.bar(layer_tags, final_hwa, color="skyblue")
    plt.ylabel("Final Validation HWA")
    plt.title("SPR_BENCH: Final HWA by # LSTM Layers")
    plt.xticks(rotation=45)
    save_path = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()
