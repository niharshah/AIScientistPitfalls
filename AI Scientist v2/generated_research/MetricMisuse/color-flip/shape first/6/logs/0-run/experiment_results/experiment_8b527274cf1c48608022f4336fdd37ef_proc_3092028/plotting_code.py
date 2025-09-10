import matplotlib.pyplot as plt
import numpy as np
import os

# working directory setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    layers_dict = experiment_data.get("GRU_NUM_LAYERS", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    layers_dict = {}


# helper to make consistent labels
def layer_label(key):  # key like 'layers_1'
    return f"{key.split('_')[1]}-layer"


# 1) Loss curves --------------------------------------------------------------
try:
    plt.figure()
    for key, rec in layers_dict.items():
        epochs = np.arange(1, len(rec["losses"]["train"]) + 1)
        plt.plot(
            epochs, rec["losses"]["train"], "--", label=f"Train {layer_label(key)}"
        )
        plt.plot(epochs, rec["losses"]["val"], "-", label=f"Val {layer_label(key)}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Toy SPR: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "toy_loss_curves_layers_compare.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Accuracy curves ----------------------------------------------------------
try:
    plt.figure()
    for key, rec in layers_dict.items():
        epochs = np.arange(1, len(rec["metrics"]["train"]) + 1)
        plt.plot(
            epochs, rec["metrics"]["train"], "--", label=f"SWA Train {layer_label(key)}"
        )
        plt.plot(
            epochs, rec["metrics"]["val"], "-", label=f"CWA Val {layer_label(key)}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Toy SPR: SWA (Train) & CWA (Val)")
    plt.legend()
    fname = os.path.join(working_dir, "toy_accuracy_curves_layers_compare.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 3) AIS curves ---------------------------------------------------------------
try:
    plt.figure()
    for key, rec in layers_dict.items():
        epochs = np.arange(1, len(rec["AIS"]["val"]) + 1)
        plt.plot(epochs, rec["AIS"]["val"], label=f"AIS Val {layer_label(key)}")
    plt.xlabel("Epoch")
    plt.ylabel("AIS")
    plt.title("Toy SPR: Agreement-Invariance Score (Validation)")
    plt.legend()
    fname = os.path.join(working_dir, "toy_AIS_curves_layers_compare.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating AIS plot: {e}")
    plt.close()

# 4) Final-epoch summary bar chart -------------------------------------------
try:
    plt.figure()
    keys = list(layers_dict.keys())
    x = np.arange(len(keys))
    width = 0.35
    cwa_final = [layers_dict[k]["metrics"]["val"][-1] for k in keys]
    ais_final = [layers_dict[k]["AIS"]["val"][-1] for k in keys]
    plt.bar(x - width / 2, cwa_final, width, label="Final CWA")
    plt.bar(x + width / 2, ais_final, width, label="Final AIS")
    plt.xticks(x, [layer_label(k) for k in keys])
    plt.ylabel("Score")
    plt.title("Toy SPR: Final Validation Scores by GRU Depth")
    plt.legend()
    fname = os.path.join(working_dir, "toy_final_score_comparison.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final summary plot: {e}")
    plt.close()
