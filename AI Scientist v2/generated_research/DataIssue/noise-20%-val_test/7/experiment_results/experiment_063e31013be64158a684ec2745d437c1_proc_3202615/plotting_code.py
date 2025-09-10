import matplotlib.pyplot as plt
import numpy as np
import os

# set up results directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    store = experiment_data["EPOCHS"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    store = None

if store:
    epochs_runs = store["epochs_list"]
    train_acc_runs = store["metrics"]["train_acc"]
    val_acc_runs = store["metrics"]["val_acc"]
    train_loss_runs = store["losses"]["train"]
    val_loss_runs = store["metrics"]["val_loss"]

    for idx, num_ep in enumerate(epochs_runs):
        # -------- Accuracy figure --------
        try:
            plt.figure()
            ep_axis = list(range(1, num_ep + 1))
            plt.plot(ep_axis, train_acc_runs[idx], label="Train Acc")
            plt.plot(ep_axis, val_acc_runs[idx], label="Val Acc")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"SPR_BENCH Accuracy Curve – {num_ep} Epochs")
            plt.legend()
            fname = f"spr_bench_acc_{str(num_ep).zfill(2)}ep.png"
            plt.savefig(os.path.join(working_dir, fname))
            print(f"Saved {fname}")
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {num_ep} epochs: {e}")
            plt.close()

        # -------- Loss figure --------
        try:
            plt.figure()
            ep_axis = list(range(1, num_ep + 1))
            plt.plot(ep_axis, train_loss_runs[idx], label="Train Loss")
            plt.plot(ep_axis, val_loss_runs[idx], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"SPR_BENCH Loss Curve – {num_ep} Epochs")
            plt.legend()
            fname = f"spr_bench_loss_{str(num_ep).zfill(2)}ep.png"
            plt.savefig(os.path.join(working_dir, fname))
            print(f"Saved {fname}")
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {num_ep} epochs: {e}")
            plt.close()
