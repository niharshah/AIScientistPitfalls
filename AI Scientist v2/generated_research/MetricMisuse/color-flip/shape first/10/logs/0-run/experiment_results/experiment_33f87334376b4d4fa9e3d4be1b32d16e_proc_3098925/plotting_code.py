import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("EPOCHS", {}).get("SPR_BENCH", {})


# ---------- helper to get metrics ----------
def get_metric(run_dict, key):
    return np.array(run_dict["metrics"].get(key, []))


# ---------- 1) Loss curves ----------
try:
    fig, axes = plt.subplots(
        len(runs), 1, figsize=(6, 3 * max(len(runs), 1)), squeeze=False
    )
    axes = axes.flatten()
    for idx, (run_name, run_dict) in enumerate(
        sorted(runs.items(), key=lambda x: int(x[0].split("_")[0]))
    ):
        tr = get_metric(run_dict, "train_loss")
        vl = get_metric(run_dict, "val_loss")
        epochs = np.arange(1, len(tr) + 1)
        axes[idx].plot(epochs, tr, label="Train")
        axes[idx].plot(epochs, vl, label="Validation")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Loss")
        axes[idx].set_title(f"SPR_BENCH Loss Curves ({run_name})")
        axes[idx].legend()
    fig.tight_layout()
    save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# ---------- 2) HWA curves ----------
try:
    fig, axes = plt.subplots(
        len(runs), 1, figsize=(6, 3 * max(len(runs), 1)), squeeze=False
    )
    axes = axes.flatten()
    for idx, (run_name, run_dict) in enumerate(
        sorted(runs.items(), key=lambda x: int(x[0].split("_")[0]))
    ):
        hwa = get_metric(run_dict, "HWA")
        epochs = np.arange(1, len(hwa) + 1)
        axes[idx].plot(epochs, hwa, color="green")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("HWA")
        axes[idx].set_title(f"SPR_BENCH HWA Curves ({run_name})")
    fig.tight_layout()
    save_path = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve figure: {e}")
    plt.close()

# ---------- 3) Final HWA comparison ----------
try:
    run_names = []
    final_hwa = []
    for run_name, run_dict in sorted(
        runs.items(), key=lambda x: int(x[0].split("_")[0])
    ):
        hwa = get_metric(run_dict, "HWA")
        if len(hwa):
            run_names.append(run_name)
            final_hwa.append(hwa[-1])
    plt.figure(figsize=(8, 4))
    plt.bar(run_names, final_hwa, color="orange")
    plt.xlabel("Epoch Configuration")
    plt.ylabel("Final HWA")
    plt.title("SPR_BENCH Final Harmonic Weighted Accuracy by Epoch Cap")
    plt.tight_layout()
    save_path = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar figure: {e}")
    plt.close()
