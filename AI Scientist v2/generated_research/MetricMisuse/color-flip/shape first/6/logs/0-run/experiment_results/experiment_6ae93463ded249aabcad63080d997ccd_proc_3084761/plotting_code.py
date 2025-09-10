import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

saved_files = []

if experiment_data is not None:
    spr = experiment_data["EPOCH_PRE_tuning"]["SPR"]
    epochs_pre_vals = spr["EPOCH_PRE_values"]
    train_losses_nested = spr["losses"][
        "train"
    ]  # list of lists (pre+ft per experiment)
    swa_nested = spr["metrics"]["train"]  # SWA per ft epoch
    cwa_nested = spr["metrics"]["val"]  # CWA per ft epoch
    ais_vals = spr["AIS"]["val"]  # final AIS

    # 1) Loss curves
    try:
        plt.figure()
        for ep_pre, losses in zip(epochs_pre_vals, train_losses_nested):
            x = range(1, len(losses) + 1)
            plt.plot(x, losses, label=f"EPOCH_PRE={ep_pre}")
        plt.xlabel("Training Step (Pre+FT Epochs)")
        plt.ylabel("Loss")
        plt.title("SPR: Training Loss Curves\nLeft: Pre-training, Right: Fine-tuning")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curves.png")
        plt.savefig(fname)
        saved_files.append(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2) SWA curves
    try:
        plt.figure()
        for ep_pre, swa in zip(epochs_pre_vals, swa_nested):
            x = range(1, len(swa) + 1)
            plt.plot(x, swa, marker="o", label=f"EPOCH_PRE={ep_pre}")
        plt.xlabel("Fine-tuning Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR: SWA over Fine-tuning Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_SWA_curves.png")
        plt.savefig(fname)
        saved_files.append(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # 3) CWA curves
    try:
        plt.figure()
        for ep_pre, cwa in zip(epochs_pre_vals, cwa_nested):
            x = range(1, len(cwa) + 1)
            plt.plot(x, cwa, marker="s", label=f"EPOCH_PRE={ep_pre}")
        plt.xlabel("Fine-tuning Epoch")
        plt.ylabel("Color-Weighted Accuracy")
        plt.title("SPR: CWA over Fine-tuning Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_CWA_curves.png")
        plt.savefig(fname)
        saved_files.append(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CWA plot: {e}")
        plt.close()

    # 4) AIS bar chart
    try:
        plt.figure()
        plt.bar([str(e) for e in epochs_pre_vals], ais_vals, color="steelblue")
        plt.xlabel("EPOCH_PRE")
        plt.ylabel("AIS (Validation)")
        plt.title("SPR: Final AIS vs Pre-training Epochs")
        fname = os.path.join(working_dir, "SPR_AIS_bar.png")
        plt.savefig(fname)
        saved_files.append(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating AIS plot: {e}")
        plt.close()

print("Saved figures:", [os.path.basename(f) for f in saved_files])
