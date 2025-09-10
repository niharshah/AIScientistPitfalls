import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    data_root = experiment_data.get("dropout_rate_tuning", {}).get("SPR_BENCH", {})
    rates = sorted(data_root.keys(), key=lambda r: float(r))
    final_dwa = {}

    # --------- Fig 1: val-loss curves ----------
    try:
        plt.figure()
        for r in rates:
            epochs = list(range(1, len(data_root[r]["losses"]["val"]) + 1))
            vloss = [v for _, v in data_root[r]["losses"]["val"]]
            plt.plot(epochs, vloss, marker="o", label=f"dropout={r}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("SPR_BENCH: Validation Loss vs Epoch for Different Dropout Rates")
        plt.legend()
        fname1 = os.path.join(working_dir, "SPR_BENCH_val_loss_curves.png")
        plt.savefig(fname1)
        plt.close()
    except Exception as e:
        print(f"Error creating validation-loss plot: {e}")
        plt.close()

    # --------- Fig 2: final DWA per rate ----------
    try:
        for r in rates:
            dwa_vals = [d for _, d in data_root[r]["metrics"]["val"] if d is not None]
            final_dwa[r] = dwa_vals[-1] if dwa_vals else 0.0
        plt.figure()
        plt.bar(list(map(float, rates)), [final_dwa[r] for r in rates], width=0.04)
        plt.xlabel("Dropout Rate")
        plt.ylabel("Final Dual-Weighted Accuracy")
        plt.title("SPR_BENCH: Final Validation DWA by Dropout Rate")
        fname2 = os.path.join(working_dir, "SPR_BENCH_final_DWA_vs_dropout.png")
        plt.savefig(fname2)
        plt.close()
    except Exception as e:
        print(f"Error creating DWA plot: {e}")
        plt.close()

    # ---------- print best setting ----------
    if final_dwa:
        best_rate = max(final_dwa, key=final_dwa.get)
        print(f"Best dropout rate: {best_rate} | DWA={final_dwa[best_rate]:.4f}")
