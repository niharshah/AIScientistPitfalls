import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
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

dt_key = "dropout_tuning"
if dt_key not in experiment_data:
    print("No dropout_tuning data found.")
    exit()


# helper to sort by numeric dropout value
def _dr(key):
    try:
        return float(key.split("_")[-1])
    except Exception:
        return 0.0


drop_keys = sorted(experiment_data[dt_key].keys(), key=_dr)

# collect arrays
epochs_dict, train_loss, val_loss, val_scwa, test_scwa = {}, {}, {}, {}, {}
for k in drop_keys:
    rec = experiment_data[dt_key][k]
    epochs_dict[k] = np.array(rec["epochs"])
    train_loss[k] = np.array(rec["losses"]["train"])
    val_loss[k] = np.array(rec["losses"]["val"])
    val_scwa[k] = np.array(rec["metrics"]["val"])
    test_scwa[k] = rec["metrics"]["test_SCWA"]

dataset_name = "synthetic_SPR"  # default
# infer dataset type from key presence; not strictly necessary
if experiment_data.get("dataset_name_1"):
    dataset_name = "real_SPR"

# ---------- plot 1: loss curves ----------
try:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for k in drop_keys:
        axs[0].plot(epochs_dict[k], train_loss[k], label=k)
        axs[1].plot(epochs_dict[k], val_loss[k], label=k)
    axs[0].set_title("Training Loss")
    axs[1].set_title("Validation Loss")
    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy")
        ax.legend()
    fig.suptitle(
        f"Loss Curves Across Dropout Rates ({dataset_name})\nLeft: Training, Right: Validation"
    )
    fp = os.path.join(working_dir, f"{dataset_name}_loss_curves.png")
    plt.savefig(fp)
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- plot 2: validation SCWA ----------
try:
    plt.figure(figsize=(6, 5))
    for k in drop_keys:
        plt.plot(epochs_dict[k], val_scwa[k], label=k)
    plt.title(f"Validation SCWA vs Epoch ({dataset_name})")
    plt.xlabel("Epoch")
    plt.ylabel("SCWA")
    plt.legend()
    fp = os.path.join(working_dir, f"{dataset_name}_val_scwa.png")
    plt.savefig(fp)
    plt.close()
except Exception as e:
    print(f"Error creating validation SCWA plot: {e}")
    plt.close()

# ---------- plot 3: test SCWA bar chart ----------
try:
    plt.figure(figsize=(6, 5))
    dps = [_dr(k) for k in drop_keys]
    scores = [test_scwa[k] for k in drop_keys]
    plt.bar(range(len(drop_keys)), scores, tick_label=[str(d) for d in dps])
    plt.title(f"Test SCWA by Dropout Rate ({dataset_name})")
    plt.xlabel("Dropout Rate")
    plt.ylabel("SCWA")
    fp = os.path.join(working_dir, f"{dataset_name}_test_scwa_bar.png")
    plt.savefig(fp)
    plt.close()
except Exception as e:
    print(f"Error creating test SCWA bar plot: {e}")
    plt.close()

# ---------- console summary ----------
print("\n=== Test SCWA Summary ===")
for k in drop_keys:
    print(f"{k}: {test_scwa[k]:.4f}")
