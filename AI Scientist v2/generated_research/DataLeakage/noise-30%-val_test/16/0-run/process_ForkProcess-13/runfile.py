import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths / loading ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ds_name = "SPR_BENCH"
    sweep = experiment_data["NUM_LSTM_LAYERS"][ds_name]
    tags = sorted(
        sweep.keys(), key=lambda x: int(x.split("_")[0])
    )  # 1_layer, 2_layer, ...

    # ---------- plot 1: MCC curves ----------
    try:
        plt.figure()
        for tag in tags:
            epochs = sweep[tag]["epochs"]
            plt.plot(
                epochs, sweep[tag]["metrics"]["train_MCC"], "--", label=f"{tag} train"
            )
            plt.plot(epochs, sweep[tag]["metrics"]["val_MCC"], "-", label=f"{tag} val")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title(f"{ds_name}: Train vs Val MCC Across Epochs")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name.lower()}_mcc_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating MCC plot: {e}")
        plt.close()

    # ---------- plot 2: Loss curves ----------
    try:
        plt.figure()
        for tag in tags:
            epochs = sweep[tag]["epochs"]
            plt.plot(epochs, sweep[tag]["losses"]["train"], "--", label=f"{tag} train")
            plt.plot(epochs, sweep[tag]["losses"]["val"], "-", label=f"{tag} val")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title(f"{ds_name}: Train vs Val Loss Across Epochs")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name.lower()}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- plot 3: Test MCC bar chart ----------
    try:
        plt.figure()
        test_mccs = [sweep[tag]["test_MCC"] for tag in tags]
        x_pos = np.arange(len(tags))
        plt.bar(x_pos, test_mccs, tick_label=tags)
        plt.ylabel("Test MCC")
        plt.title(f"{ds_name}: Test MCC by #LSTM Layers")
        fname = os.path.join(working_dir, f"{ds_name.lower()}_test_mcc_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test MCC bar plot: {e}")
        plt.close()

    # ---------- console summary ----------
    print("\nTest MCC scores:")
    for tag, mcc in zip(tags, test_mccs):
        print(f"  {tag}: {mcc:.4f}")
