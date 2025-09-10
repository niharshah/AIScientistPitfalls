import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["EPOCH_TUNING"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    epochs = [hp["epochs"] for hp in ed["hyperparams"]]
    val_acc, test_acc = ed["metrics"]["val_acc"], ed["metrics"]["test_acc"]
    val_ura, test_ura = ed["metrics"]["val_ura"], ed["metrics"]["test_ura"]
    train_losses = ed["losses"]["train"]

    # ---------- plot 1: accuracy ----------
    try:
        plt.figure()
        plt.plot(epochs, val_acc, "o-", label="Validation")
        plt.plot(epochs, test_acc, "s-", label="Test")
        plt.xlabel("Epoch Budget")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Accuracy across Epoch Budgets\nValidation vs Test")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_vs_epoch.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------- plot 2: URA ----------
    try:
        plt.figure()
        plt.plot(epochs, val_ura, "o-", label="Validation")
        plt.plot(epochs, test_ura, "s-", label="Test")
        plt.xlabel("Epoch Budget")
        plt.ylabel("URA")
        plt.title("SPR_BENCH: URA across Epoch Budgets\nValidation vs Test")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_URA_vs_epoch.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating URA plot: {e}")
        plt.close()

    # ---------- plot 3: loss curves (≤5) ----------
    try:
        idxs = np.linspace(0, len(epochs) - 1, min(5, len(epochs)), dtype=int)
        plt.figure()
        for i in idxs:
            plt.plot(
                range(1, len(train_losses[i]) + 1),
                train_losses[i],
                label=f"{epochs[i]} epochs",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title("SPR_BENCH: Training Loss Curves\nSelected Epoch Budgets")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- console summary ----------
    print("Epoch Budget | Test Acc | Test URA")
    for e, a, u in zip(epochs, test_acc, test_ura):
        print(f"{e:>12} | {a:.3f}    | {u:.3f}")
