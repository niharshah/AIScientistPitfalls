import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------
try:
    spr_exp = experiment_data["epochs"]["SPR_BENCH"]
except Exception as e:
    print(f"SPR_BENCH data not found: {e}")
    spr_exp = {}

# --------- plot 1-4: loss curves for each epoch budget -------------
for i, (ep_str, res) in enumerate(sorted(spr_exp.items(), key=lambda x: int(x[0]))):
    try:
        train_loss = res["losses"]["train"]
        val_loss = res["losses"]["val"]
        epochs = range(1, len(train_loss) + 1)

        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH Train vs Val Loss ({ep_str} Epochs)")
        plt.legend()
        fname = f"SPR_BENCH_loss_curves_{ep_str}epochs.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ep_str} epochs: {e}")
        plt.close()

# --------- plot 5: aggregated CWA curves ---------------------------
try:
    plt.figure()
    for ep_str, res in sorted(spr_exp.items(), key=lambda x: int(x[0])):
        cwa = res["metrics"]["val_cwa2"]
        epochs = range(1, len(cwa) + 1)
        plt.plot(epochs, cwa, label=f"{ep_str} Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.title("SPR_BENCH Validation CWA Across Models")
    plt.legend()
    fname = "SPR_BENCH_val_CWA_comparison.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated CWA plot: {e}")
    plt.close()
