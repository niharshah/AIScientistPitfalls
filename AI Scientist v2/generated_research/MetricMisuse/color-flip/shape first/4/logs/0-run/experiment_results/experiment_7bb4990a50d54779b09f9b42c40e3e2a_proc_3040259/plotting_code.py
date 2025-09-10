import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


def get_runs(setting):
    return experiment_data.get(setting, {}).get("SPR_BENCH", {})


baseline_runs = get_runs("baseline")
rand_runs = get_runs("token_order_randomization")


# helper to extract per-run arrays
def extract_curves(runs, key):
    out = {}
    for epochs, rec in runs.items():
        out[int(epochs)] = rec[key]
    return out  # {num_epochs: list}


# ------------------ plotting ------------------
# 1) Baseline losses
try:
    curves_tr = (
        extract_curves(baseline_runs, "losses")["train"] if False else None
    )  # placeholder to trigger except if not exist
except Exception:
    pass  # dummy so that pylint doesn't complain about undefined
try:
    loss_dict = extract_curves(baseline_runs, "losses")
    if loss_dict:
        plt.figure()
        for ep, losses in loss_dict.items():
            plt.plot(
                range(1, len(losses["train"]) + 1),
                losses["train"],
                label=f"{ep}ep-train",
            )
            plt.plot(
                range(1, len(losses["val"]) + 1),
                losses["val"],
                linestyle="--",
                label=f"{ep}ep-val",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Baseline SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_baseline_losses.png"))
        plt.close()
except Exception as e:
    print(f"Error creating Baseline Loss plot: {e}")
    plt.close()

# 2) Baseline HWA
try:
    hwa_dict = {int(ep): rec["metrics"]["val"] for ep, rec in baseline_runs.items()}
    if hwa_dict:
        plt.figure()
        for ep, hwa in hwa_dict.items():
            plt.plot(range(1, len(hwa) + 1), hwa, label=f"{ep} epochs")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("Baseline SPR_BENCH: Harmonic-Weighted Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_baseline_hwa.png"))
        plt.close()
except Exception as e:
    print(f"Error creating Baseline HWA plot: {e}")
    plt.close()

# 3) Randomization losses
try:
    loss_dict = extract_curves(rand_runs, "losses")
    if loss_dict:
        plt.figure()
        for ep, losses in loss_dict.items():
            plt.plot(
                range(1, len(losses["train"]) + 1),
                losses["train"],
                label=f"{ep}ep-train",
            )
            plt.plot(
                range(1, len(losses["val"]) + 1),
                losses["val"],
                linestyle="--",
                label=f"{ep}ep-val",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Token-Order Randomization SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_randomization_losses.png"))
        plt.close()
except Exception as e:
    print(f"Error creating Randomization Loss plot: {e}")
    plt.close()

# 4) Randomization HWA
try:
    hwa_dict = {int(ep): rec["metrics"]["val"] for ep, rec in rand_runs.items()}
    if hwa_dict:
        plt.figure()
        for ep, hwa in hwa_dict.items():
            plt.plot(range(1, len(hwa) + 1), hwa, label=f"{ep} epochs")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("Token-Order Randomization SPR_BENCH: Harmonic-Weighted Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_randomization_hwa.png"))
        plt.close()
except Exception as e:
    print(f"Error creating Randomization HWA plot: {e}")
    plt.close()

# 5) Final-epoch HWA comparison
try:
    if baseline_runs and rand_runs:
        epochs_sorted = sorted(
            set(int(e) for e in baseline_runs) & set(int(e) for e in rand_runs)
        )
        base_final = [
            baseline_runs[str(e)]["metrics"]["val"][-1] for e in epochs_sorted
        ]
        rand_final = [rand_runs[str(e)]["metrics"]["val"][-1] for e in epochs_sorted]
        plt.figure()
        plt.plot(epochs_sorted, base_final, marker="o", label="Baseline")
        plt.plot(epochs_sorted, rand_final, marker="s", label="Token-Order Rand.")
        plt.xlabel("Training Epochs")
        plt.ylabel("Final HWA")
        plt.title("SPR_BENCH: Final HWA vs Epochs (Baseline vs Token-Order Rand.)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_hwa_comparison.png"))
        plt.close()
except Exception as e:
    print(f"Error creating HWA comparison plot: {e}")
    plt.close()
