import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load experiment data ----------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_runs = experiment_data.get("HIDDEN_DIM_TUNING", {}).get("SPR_BENCH", {})

# ----------------- individual loss curves ---------------- #
for hid_str, run in spr_runs.items():
    try:
        epochs = run["epochs"]
        train_loss = run["losses"]["train"]
        val_loss = run["losses"]["val"]

        plt.figure()
        plt.plot(epochs, train_loss, label="train")
        plt.plot(epochs, val_loss, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"SPR_BENCH – Loss Curves (hid={hid_str})")
        plt.legend()
        fname = f"SPR_BENCH_loss_curve_hid{hid_str}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for hid={hid_str}: {e}")
        plt.close()

# ----------------- final test MCC bar plot ---------------- #
try:
    hids_sorted = sorted(spr_runs.keys(), key=int)
    test_mccs = [spr_runs[h]["metrics"]["test_MCC"] for h in hids_sorted]

    plt.figure()
    plt.bar(hids_sorted, test_mccs)
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Test MCC")
    plt.title("SPR_BENCH – Final Test MCC vs Hidden Dim")
    fname = "SPR_BENCH_test_MCC_comparison.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating test MCC bar plot: {e}")
    plt.close()
