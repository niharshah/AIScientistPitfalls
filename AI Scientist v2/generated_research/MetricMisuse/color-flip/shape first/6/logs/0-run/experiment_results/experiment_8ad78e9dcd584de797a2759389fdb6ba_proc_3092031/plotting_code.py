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


# helper to fetch safely
def get(d, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default


spr_data = get(experiment_data, "contrastive_temperature", "SPR", default={})
temps_sorted = sorted(spr_data.keys())

ais_summary = []

# ----------- plotting -------------
for idx, temp in enumerate(temps_sorted):
    try:
        d = spr_data[temp]
        tr_loss = d["losses"]["train"]
        val_loss = d["losses"]["val"]
        swa = d["metrics"]["train"]
        cwa = d["metrics"]["val"]
        ais = d["AIS"]["val"]
        epochs = range(1, len(tr_loss) + 1)

        plt.figure(figsize=(10, 4))
        # -------- left subplot : losses ----------
        plt.subplot(1, 2, 1)
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.title("Loss Curves")
        plt.legend()

        # -------- right subplot : metrics ----------
        plt.subplot(1, 2, 2)
        if swa:
            plt.plot(epochs, swa, label="SWA (train)")
        if cwa:
            plt.plot(epochs, cwa, label="CWA (val)")
        if ais:
            plt.plot(epochs, ais, label="AIS (val)")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Metric Curves")
        plt.legend()

        plt.suptitle(
            f"SPR Training Curves\nLeft: Loss   |   Right: Metrics  (temperature={temp})"
        )
        fname = f"SPR_temp{temp}_training_curves.png"
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()

    except Exception as e:
        print(f"Error creating plot for temperature {temp}: {e}")
        plt.close()

    # collect final AIS
    if ais:
        ais_summary.append((temp, ais[-1]))

# ------------ print AIS table -------------
if ais_summary:
    print("Final validation AIS per temperature:")
    for t, v in ais_summary:
        print(f"  temp={t:.1f} -> AIS={v:.3f}")
