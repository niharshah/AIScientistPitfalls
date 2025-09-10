import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# quick sanity check
branch = experiment_data.get("uni_directional_encoder", {}).get("SPR_BENCH", {})
if not branch:
    print("No data found for uni_directional_encoder / SPR_BENCH")
    exit()

# gather metrics
epochs_list, swa_vals, cwa_vals, schm_vals = [], [], [], []
for ep_str, run_dict in sorted(branch.items(), key=lambda x: int(x[0])):
    epochs_list.append(int(ep_str))
    swa_vals.append(run_dict["metrics"]["SWA"][-1])
    cwa_vals.append(run_dict["metrics"]["CWA"][-1])
    schm_vals.append(run_dict["metrics"]["SCHM"][-1])

print("Pretrain epochs:", epochs_list)
print("SWA:", swa_vals)
print("CWA:", cwa_vals)
print("SCHM:", schm_vals)


# plotting helper
def _plot(x, y, metric_name):
    try:
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.title(f"{metric_name} vs Pre-training Epochs – SPR_BENCH")
        plt.xlabel("Number of Pre-training Epochs")
        plt.ylabel(metric_name)
        plt.grid(True, alpha=0.3)
        fname = os.path.join(
            working_dir, f"SPR_BENCH_{metric_name.lower()}_vs_pretrain_epochs.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {metric_name}: {e}")
        plt.close()


# create three plots
_plot(epochs_list, swa_vals, "SWA")
_plot(epochs_list, cwa_vals, "CWA")
_plot(epochs_list, schm_vals, "SCHM")
