import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to safely fetch nested dict
def get_ed(data):
    try:
        return data["multi_synth_generalization"]["D1-D2-D3"]
    except KeyError:
        return None


ed = get_ed(experiment_data)

if ed is not None:
    # extract series
    train_losses = ed["losses"]["train"]
    val_losses = ed["losses"]["val"]
    train_accs = [m["acc"] for m in ed["metrics"]["train"]]
    val_accs = [m["acc"] for m in ed["metrics"]["val"]]
    train_swa = [m["swa"] for m in ed["metrics"]["train"]]
    val_swa = [m["swa"] for m in ed["metrics"]["val"]]

    # 1. Loss curves ---------------------------------------------------------
    try:
        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Loss Curves – multi_synth_generalization")
        plt.legend()
        fname = os.path.join(working_dir, "multi_synth_generalization_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2. Accuracy curves -----------------------------------------------------
    try:
        plt.figure()
        plt.plot(train_accs, label="Train")
        plt.plot(val_accs, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curves – multi_synth_generalization")
        plt.legend()
        fname = os.path.join(
            working_dir, "multi_synth_generalization_accuracy_curves.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 3. Shape-Weighted Accuracy curves --------------------------------------
    try:
        plt.figure()
        plt.plot(train_swa, label="Train")
        plt.plot(val_swa, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("Shape-Weighted Accuracy Curves – multi_synth_generalization")
        plt.legend()
        fname = os.path.join(working_dir, "multi_synth_generalization_swa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    print("Plots saved to:", working_dir)
else:
    print("Required experiment entry not found.")
