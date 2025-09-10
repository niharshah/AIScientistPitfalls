import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------ load data ------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["hidden_dim_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

# Keys & helpers --------------------------------------------------------
hidden_dims = sorted([int(k.split("_")[-1]) for k in exp.keys()])
epochs = list(range(1, 1 + len(next(iter(exp.values()))["metrics"]["train_acc"])))

# ------------------------ FIGURE 1 : accuracy curves ------------------
try:
    plt.figure()
    for k in sorted(exp.keys(), key=lambda x: int(x.split("_")[-1])):
        hd = int(k.split("_")[-1])
        tr_acc = exp[k]["metrics"]["train_acc"]
        val_acc = exp[k]["metrics"]["val_acc"]
        plt.plot(epochs, tr_acc, marker="o", label=f"train hd{hd}")
        plt.plot(epochs, val_acc, marker="x", linestyle="--", label=f"val hd{hd}")
    plt.title("SPR_BENCH: Train & Val Accuracy vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_hidden_dim_accuracy_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# ------------------------ FIGURE 2 : loss curves ----------------------
try:
    plt.figure()
    for k in sorted(exp.keys(), key=lambda x: int(x.split("_")[-1])):
        hd = int(k.split("_")[-1])
        tr_loss = exp[k]["losses"]["train"]
        val_loss = exp[k]["losses"]["val"]
        plt.plot(epochs, tr_loss, marker="o", label=f"train hd{hd}")
        plt.plot(epochs, val_loss, marker="x", linestyle="--", label=f"val hd{hd}")
    plt.title("SPR_BENCH: Train & Val Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_hidden_dim_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------ FIGURE 3 : final val accuracy ---------------
try:
    plt.figure()
    finals = [exp[f"hidden_{hd}"]["metrics"]["val_acc"][-1] for hd in hidden_dims]
    plt.bar([str(hd) for hd in hidden_dims], finals, color="skyblue")
    plt.title("SPR_BENCH: Final Validation Accuracy by Hidden Dim")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Validation Accuracy (Epoch 5)")
    fname = os.path.join(working_dir, "SPR_BENCH_final_val_accuracy_bar.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating final val accuracy bar: {e}")
    plt.close()

# ------------------------ FIGURE 4 : ZSRTA bar ------------------------
try:
    plt.figure()
    zsrtas = [exp[f"hidden_{hd}"]["metrics"]["ZSRTA"][0] for hd in hidden_dims]
    plt.bar([str(hd) for hd in hidden_dims], zsrtas, color="salmon")
    plt.title("SPR_BENCH: Zero-Shot Rule Transfer Accuracy (ZSRTA)")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("ZSRTA")
    fname = os.path.join(working_dir, "SPR_BENCH_ZSRTA_bar.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating ZSRTA bar: {e}")
    plt.close()
