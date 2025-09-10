import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- ENV --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- LOAD DATA --------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
variants = list(experiment_data.keys())  # ['learned_positional_embeddings', ...]
epochs = range(
    1,
    (
        1
        + max(
            len(v["metrics"]["val_SCWA"])
            for v in experiment_data[variants[0]][dataset].values()
        )
        if experiment_data
        else 0
    ),
)

# -------------------- PLOTS ------------------
# 1) Fine-tuning losses
try:
    plt.figure()
    for var in variants:
        d = experiment_data[var][dataset]
        plt.plot(d["losses"]["train"], label=f"{var}-train")
        plt.plot(d["losses"]["val"], label=f"{var}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{dataset}: Fine-tuning Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{dataset}_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating fine-tuning loss plot: {e}")
    plt.close()

# 2) Pre-training loss
try:
    plt.figure()
    for var in variants:
        d = experiment_data[var][dataset]
        plt.plot(d["losses"]["pretrain"], label=f"{var}-pretrain")
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent Loss")
    plt.title(f"{dataset}: Pre-training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{dataset}_pretrain_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating pre-training loss plot: {e}")
    plt.close()

# 3) Validation SCWA
try:
    plt.figure()
    for var in variants:
        d = experiment_data[var][dataset]
        plt.plot(d["metrics"]["val_SCWA"], label=var)
    plt.xlabel("Epoch")
    plt.ylabel("SCWA")
    plt.title(f"{dataset}: Validation SCWA across Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{dataset}_SCWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SCWA plot: {e}")
    plt.close()

# 4) SWA & CWA side-by-side
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for var in variants:
        d = experiment_data[var][dataset]
        axes[0].plot(d["metrics"]["val_SWA"], label=var)
        axes[1].plot(d["metrics"]["val_CWA"], label=var)
    axes[0].set_title("Left: SWA")
    axes[1].set_title("Right: CWA")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
    axes[0].set_ylabel("Weighted Accuracy")
    fig.suptitle(f"{dataset}: Validation Weighted Accuracies")
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig_path = os.path.join(working_dir, f"{dataset}_SWA_CWA.png")
    plt.savefig(fig_path)
    plt.close()
except Exception as e:
    print(f"Error creating SWA/CWA plot: {e}")
    plt.close()

# -------------------- PRINT BEST SCWA --------
for var in variants:
    try:
        best_scwa = max(experiment_data[var][dataset]["metrics"]["val_SCWA"])
        print(f"{var} best validation SCWA: {best_scwa:.4f}")
    except Exception as e:
        print(f"Could not compute best SCWA for {var}: {e}")
