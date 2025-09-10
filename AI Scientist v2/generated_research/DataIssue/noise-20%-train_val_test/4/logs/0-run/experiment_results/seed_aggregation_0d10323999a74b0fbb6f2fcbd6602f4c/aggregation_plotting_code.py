import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt

# ---------------------------------------------------------------
# setup + load all experiment runs
# ---------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2025-08-17_00-44-46_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_c3db80386b7b4727b10aae88815ac644_proc_3167391/experiment_data.npy",
    "experiments/2025-08-17_00-44-46_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_3da980b1b46f4371a34515ff1084cc79_proc_3167394/experiment_data.npy",
    "experiments/2025-08-17_00-44-46_contextual_embedding_spr_attempt_0/logs/0-run/experiment_results/experiment_d614983917594c91b463acb196106c65_proc_3167393/experiment_data.npy",
]

all_runs = []
try:
    for p in experiment_data_path_list:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        data = np.load(full_p, allow_pickle=True).item()
        all_runs.append(data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_runs = []

# ---------------------------------------------------------------
# aggregate curves
# ---------------------------------------------------------------
train_losses, val_losses = [], []
train_f1s, val_f1s = [], []
test_f1s = []

for run in all_runs:
    cbc = run.get("char_bigram_count", {})
    tl = np.asarray(cbc.get("losses", {}).get("train", []))
    vl = np.asarray(cbc.get("losses", {}).get("val", []))
    tf1 = np.asarray(cbc.get("metrics", {}).get("train_f1", []))
    vf1 = np.asarray(cbc.get("metrics", {}).get("val_f1", []))
    if tl.size and vl.size and tf1.size and vf1.size:
        train_losses.append(tl)
        val_losses.append(vl)
        train_f1s.append(tf1)
        val_f1s.append(vf1)
    tf1_final = cbc.get("metrics", {}).get("test_f1", None)
    if tf1_final is not None:
        test_f1s.append(float(tf1_final))

# Make sure we have data
if not train_losses:
    print("No runs contained usable curve data; aborting plots.")
    exit()

# Trim to shortest run length so arrays align
min_len = min(map(len, train_losses))
train_losses = np.stack([a[:min_len] for a in train_losses])
val_losses = np.stack([a[:min_len] for a in val_losses])
train_f1s = np.stack([a[:min_len] for a in train_f1s])
val_f1s = np.stack([a[:min_len] for a in val_f1s])
epochs = np.arange(min_len)

n_runs = train_losses.shape[0]
sem = lambda x: np.std(x, axis=0) / sqrt(n_runs)

# ---------------------------------------------------------------
# 1) Aggregated loss curves
# ---------------------------------------------------------------
try:
    plt.figure(figsize=(10, 4))
    # Training
    plt.subplot(1, 2, 1)
    mean_tl = train_losses.mean(axis=0)
    plt.plot(epochs, mean_tl, label="train mean")
    plt.fill_between(
        epochs,
        mean_tl - sem(train_losses),
        mean_tl + sem(train_losses),
        alpha=0.3,
        label="±1 SEM",
    )
    plt.title("Left: Aggregated Training Loss - SPR_BENCH (char_bigram_count)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Validation
    plt.subplot(1, 2, 2)
    mean_vl = val_losses.mean(axis=0)
    plt.plot(epochs, mean_vl, label="val mean", color="orange")
    plt.fill_between(
        epochs,
        mean_vl - sem(val_losses),
        mean_vl + sem(val_losses),
        alpha=0.3,
        color="orange",
        label="±1 SEM",
    )
    plt.title("Right: Aggregated Validation Loss - SPR_BENCH (char_bigram_count)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    fname = os.path.join(
        working_dir, "SPR_BENCH_char_bigram_count_aggregated_loss_curves.png"
    )
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# ---------------------------------------------------------------
# 2) Aggregated F1 curves
# ---------------------------------------------------------------
try:
    plt.figure(figsize=(10, 4))
    # Training
    plt.subplot(1, 2, 1)
    mean_tf1 = train_f1s.mean(axis=0)
    plt.plot(epochs, mean_tf1, label="train mean")
    plt.fill_between(
        epochs,
        mean_tf1 - sem(train_f1s),
        mean_tf1 + sem(train_f1s),
        alpha=0.3,
        label="±1 SEM",
    )
    plt.title("Left: Aggregated Training Macro-F1 - SPR_BENCH (char_bigram_count)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()

    # Validation
    plt.subplot(1, 2, 2)
    mean_vf1 = val_f1s.mean(axis=0)
    plt.plot(epochs, mean_vf1, label="val mean", color="orange")
    plt.fill_between(
        epochs,
        mean_vf1 - sem(val_f1s),
        mean_vf1 + sem(val_f1s),
        alpha=0.3,
        color="orange",
        label="±1 SEM",
    )
    plt.title("Right: Aggregated Validation Macro-F1 - SPR_BENCH (char_bigram_count)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()

    plt.tight_layout()
    fname = os.path.join(
        working_dir, "SPR_BENCH_char_bigram_count_aggregated_f1_curves.png"
    )
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating aggregated F1 curves: {e}")
    plt.close()

# ---------------------------------------------------------------
# Print aggregated test metric
# ---------------------------------------------------------------
if test_f1s:
    mean_test = np.mean(test_f1s)
    sem_test = np.std(test_f1s) / sqrt(len(test_f1s))
    print(f"Aggregated Test Macro-F1: {mean_test:.4f} ± {sem_test:.4f}")
