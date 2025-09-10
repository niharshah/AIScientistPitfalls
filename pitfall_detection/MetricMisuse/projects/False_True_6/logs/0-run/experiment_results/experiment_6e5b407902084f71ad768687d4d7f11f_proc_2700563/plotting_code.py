import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data -------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    hid_dict = experiment_data["hidden_size_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    hid_dict = {}


# Helper to get ordered (hid, rec) list --------------------------------
def sorted_hids(dic):
    def _hid(k):  # "h_32" -> 32
        try:
            return int(k.split("_")[1])
        except:
            return k

    return sorted(dic.items(), key=lambda kv: _hid(kv[0]))


# 1) Loss curves -------------------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    for hid_key, rec in sorted_hids(hid_dict):
        tr_losses = rec["losses"]["train"]
        val_losses = rec["losses"]["val"]
        epochs = np.arange(1, len(tr_losses) + 1)
        plt.plot(epochs, tr_losses, "--", label=f"{hid_key} train")
        plt.plot(epochs, val_losses, "-", label=f"{hid_key} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss\nHidden-size sweep")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves_hidden_sweep.png")
    plt.tight_layout()
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2) Validation accuracy curves ---------------------------------------
try:
    plt.figure(figsize=(6, 4))
    for hid_key, rec in sorted_hids(hid_dict):
        val_metrics = rec["metrics"]["val"]
        accs = [m["acc"] for m in val_metrics if "acc" in m]
        epochs = np.arange(1, len(accs) + 1)
        plt.plot(epochs, accs, marker="o", label=f"{hid_key}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("SPR_BENCH: Validation Accuracy\nHidden-size sweep")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_val_accuracy_hidden_sweep.png")
    plt.tight_layout()
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating val-accuracy plot: {e}")
    plt.close()

# 3) Test accuracy bar chart ------------------------------------------
try:
    labels, test_accs = [], []
    for hid_key, rec in sorted_hids(hid_dict):
        if "test" in rec["metrics"] and "acc" in rec["metrics"]["test"]:
            labels.append(hid_key)
            test_accs.append(rec["metrics"]["test"]["acc"])
    if test_accs:
        plt.figure(figsize=(6, 4))
        plt.bar(labels, test_accs, color="orange")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Test Accuracy per Hidden Size")
        fname = os.path.join(working_dir, "spr_bench_test_accuracy_bar.png")
        plt.tight_layout()
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating test-accuracy bar chart: {e}")
    plt.close()
