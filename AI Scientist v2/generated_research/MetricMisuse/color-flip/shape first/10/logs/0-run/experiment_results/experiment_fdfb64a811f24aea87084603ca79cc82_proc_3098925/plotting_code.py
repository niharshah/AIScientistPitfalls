import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load data ------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ls_runs = experiment_data.get("label_smoothing", {})
if not ls_runs:
    raise SystemExit("No label_smoothing data found.")

alphas = sorted(float(k.split("_")[-1]) for k in ls_runs.keys())
alpha_keys = [f"alpha_{a}" for a in alphas]
epochs = range(1, len(next(iter(ls_runs.values()))["losses"]["train"]) + 1)


# helper to pull arrays
def arr(run, cat, split=None):
    d = ls_runs[run][cat]
    return d[split] if split else d


# ------------ plot 1: loss curves ------------
try:
    plt.figure(figsize=(6, 4))
    for ak in alpha_keys:
        plt.plot(epochs, arr(ak, "losses", "train"), label=f"train α={ak[6:]}")
        plt.plot(
            epochs, arr(ak, "losses", "val"), linestyle="--", label=f"val α={ak[6:]}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Synthetic Dataset\nTrain vs Validation Loss across α")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "spr_label_smoothing_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------ plot 2: SWA & CWA over epochs ------------
try:
    plt.figure(figsize=(6, 4))
    for metric in ["SWA", "CWA"]:
        for ak in alpha_keys:
            plt.plot(
                epochs,
                arr(ak, "metrics")[metric],
                label=f"{metric} α={ak[6:]}",
                linestyle="-" if metric == "SWA" else "--",
            )
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("SPR Synthetic Dataset\nSWA and CWA across α")
    plt.legend(fontsize=7, ncol=2)
    fname = os.path.join(working_dir, "spr_label_smoothing_weighted_acc.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating weighted accuracy plot: {e}")
    plt.close()

# ------------ plot 3: final HWA bar chart ------------
try:
    final_hwa = [arr(ak, "metrics")["HWA"][-1] for ak in alpha_keys]
    plt.figure(figsize=(4, 3))
    plt.bar([str(a) for a in alphas], final_hwa, color="skyblue")
    plt.xlabel("α")
    plt.ylabel("Final-Epoch HWA")
    plt.title("SPR Synthetic Dataset\nFinal Harmonic Weighted Accuracy")
    fname = os.path.join(working_dir, "spr_label_smoothing_final_HWA.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating HWA bar chart: {e}")
    plt.close()

# ------------ print final metrics table ------------
print("\nFinal-epoch metrics by α")
print("α\tSWA\tCWA\tHWA")
for ak, a in zip(alpha_keys, alphas):
    swa, cwa, hwa = (arr(ak, "metrics")[m][-1] for m in ["SWA", "CWA", "HWA"])
    print(f"{a:.2f}\t{swa:.3f}\t{cwa:.3f}\t{hwa:.3f}")
