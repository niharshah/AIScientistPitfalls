import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    runs = experiment_data["epochs_tuning"]["SPR_BENCH"]["runs"]
    colors = ["tab:blue", "tab:orange", "tab:green"]  # one per budget
    budgets = [r["epochs"] for r in runs]

    # ---------- figure 1: loss curves ----------
    try:
        plt.figure(figsize=(8, 5))
        for i, run in enumerate(runs):
            ep, tr_loss = zip(*run["losses"]["train"])
            _, val_loss = zip(*run["losses"]["val"])
            plt.plot(
                ep,
                tr_loss,
                linestyle="-",
                color=colors[i],
                label=f"{budgets[i]}e Train",
            )
            plt.plot(
                ep,
                val_loss,
                linestyle="--",
                color=colors[i],
                label=f"{budgets[i]}e Val",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Training vs Validation Loss (All Runs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- figure 2: HWA curves ----------
    try:
        plt.figure(figsize=(8, 5))
        for i, run in enumerate(runs):
            ep, _, _, hwa = zip(*run["metrics"]["val"])
            plt.plot(ep, hwa, color=colors[i], label=f"{budgets[i]} epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy")
        plt.title("SPR_BENCH – Validation HWA Across Epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_metric_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot: {e}")
        plt.close()

    # ---------- figure 3: final metrics bar chart ----------
    try:
        final_swa, final_cwa, final_hwa = [], [], []
        for run in runs:
            _, swa, cwa, hwa = run["metrics"]["val"][-1]
            final_swa.append(swa)
            final_cwa.append(cwa)
            final_hwa.append(hwa)

        x = np.arange(len(budgets))
        width = 0.25
        plt.figure(figsize=(9, 5))
        plt.bar(x - width, final_swa, width, label="SWA")
        plt.bar(x, final_cwa, width, label="CWA")
        plt.bar(x + width, final_hwa, width, label="HWA")
        plt.xticks(x, [f"{b}e" for b in budgets])
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH – Final Validation Metrics per Epoch Budget")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating final metrics plot: {e}")
        plt.close()

    # ---------- print final numbers ----------
    print("Final Validation Metrics")
    print("Budget\tSWA\tCWA\tHWA")
    for b, s, c, h in zip(budgets, final_swa, final_cwa, final_hwa):
        print(f"{b}\t{s:.4f}\t{c:.4f}\t{h:.4f}")
