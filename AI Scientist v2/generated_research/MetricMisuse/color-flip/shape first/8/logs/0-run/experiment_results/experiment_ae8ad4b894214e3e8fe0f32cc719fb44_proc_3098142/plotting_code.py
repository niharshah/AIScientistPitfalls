import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = exp["temperature_tuning"]["SPR_BENCH"]
    temps = sorted(data.keys())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data, temps = {}, []


# ---------- helper to fetch epoch series ----------
def series(key, t):
    return (
        data[t]["losses"]["train"]
        if key == "train_loss"
        else (
            data[t]["losses"]["val"]
            if key == "val_loss"
            else data[t]["metrics"]["val_SCWA"]
        )
    )


# ---------- Figure 1: loss curves ----------
try:
    plt.figure(figsize=(7, 5))
    for t in temps:
        plt.plot(series("train_loss", t), label=f"Train Loss T={t}")
        plt.plot(series("val_loss", t), linestyle="--", label=f"Val Loss T={t}")
    plt.title("SPR_BENCH: Train vs Val Loss (all temperatures)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname1 = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname1)
    print(f"Saved {fname1}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- Figure 2: SCWA per epoch ----------
try:
    plt.figure(figsize=(7, 5))
    for t in temps:
        plt.plot(series("val_SCWA", t), label=f"T={t}")
    plt.title("SPR_BENCH: Validation SCWA Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SCWA")
    plt.legend()
    fname2 = os.path.join(working_dir, "SPR_BENCH_val_SCWA_curves.png")
    plt.savefig(fname2)
    print(f"Saved {fname2}")
    plt.close()
except Exception as e:
    print(f"Error creating SCWA curve plot: {e}")
    plt.close()

# ---------- Figure 3: final SCWA bar chart ----------
try:
    final_scwa = [series("val_SCWA", t)[-1] for t in temps]
    best_idx = int(np.argmax(final_scwa))
    colors = ["tab:blue"] * len(temps)
    colors[best_idx] = "tab:orange"
    plt.figure(figsize=(6, 4))
    plt.bar([str(t) for t in temps], final_scwa, color=colors)
    plt.title("SPR_BENCH: Final-Epoch SCWA by Temperature\n(orange = best)")
    plt.xlabel("Temperature")
    plt.ylabel("Final SCWA")
    fname3 = os.path.join(working_dir, "SPR_BENCH_final_SCWA_bar.png")
    plt.savefig(fname3)
    print(f"Saved {fname3}")
    plt.close()
except Exception as e:
    print(f"Error creating final SCWA bar plot: {e}")
    plt.close()
