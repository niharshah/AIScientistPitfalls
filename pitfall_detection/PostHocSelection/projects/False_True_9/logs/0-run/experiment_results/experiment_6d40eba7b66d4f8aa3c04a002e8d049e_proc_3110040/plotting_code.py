import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------- paths & load --------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to reach inner dicts safely
def get_inner():
    try:
        model_key = next(iter(experiment_data))
        ds_key = next(iter(experiment_data[model_key]))
        sweep = experiment_data[model_key][ds_key]["hidden_size"]
        return model_key, ds_key, sweep
    except Exception as e:
        print(f"Could not parse experiment structure: {e}")
        return None, None, {}


model_key, ds_key, sweep = get_inner()

# ------------------------------- Figure 1: losses ------------------------------
try:
    plt.figure()
    for hs, store in sweep.items():
        # unpack epoch,loss
        tr = np.array(store["losses"]["train"])
        vl = np.array(store["losses"]["val"])
        if len(tr):
            plt.plot(tr[:, 0], tr[:, 1], label=f"train h{hs}")
        if len(vl):
            plt.plot(vl[:, 0], vl[:, 1], "--", label=f"val h{hs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{ds_key}: Training/Validation Loss Curves\nUniLSTM sweep")
    plt.legend(fontsize=7)
    fpath = os.path.join(working_dir, f"{ds_key}_loss_curves_Unidirectional_LSTM.png")
    plt.savefig(fpath, bbox_inches="tight")
    print(f"Saved {fpath}")
    plt.close()
except Exception as e:
    print(f"Error creating loss figure: {e}")
    plt.close()

# ------------------------- Figure 2: HWA vs epoch ------------------------------
try:
    plt.figure()
    for hs, store in sweep.items():
        met = np.array(store["metrics"]["val"])  # ep,swa,cwa,hwa
        if len(met):
            plt.plot(met[:, 0], met[:, 3], label=f"h{hs}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Acc.")
    plt.title(f"{ds_key}: Validation HWA across Epochs\nUniLSTM sweep")
    plt.legend(fontsize=7)
    fpath = os.path.join(working_dir, f"{ds_key}_HWA_curves_Unidirectional_LSTM.png")
    plt.savefig(fpath, bbox_inches="tight")
    print(f"Saved {fpath}")
    plt.close()
except Exception as e:
    print(f"Error creating HWA figure: {e}")
    plt.close()

# ------------------- Figure 3: final HWA per hidden size -----------------------
try:
    plt.figure()
    h_sizes, final_hwa = [], []
    for hs, store in sweep.items():
        met = np.array(store["metrics"]["val"])
        if len(met):
            h_sizes.append(str(hs))
            final_hwa.append(met[-1, 3])
    plt.bar(h_sizes, final_hwa, color="skyblue")
    plt.xlabel("Hidden Size")
    plt.ylabel("Final HWA")
    plt.title(f"{ds_key}: Final Harmonic Weighted Accuracy\nUniLSTM sweep")
    for i, v in enumerate(final_hwa):
        plt.text(i, v + 0.002, f"{v:.2f}", ha="center", fontsize=7)
    fpath = os.path.join(working_dir, f"{ds_key}_Final_HWA_Unidirectional_LSTM.png")
    plt.savefig(fpath, bbox_inches="tight")
    print(f"Saved {fpath}")
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar figure: {e}")
    plt.close()
