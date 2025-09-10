import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up working dir ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ---------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- figure generation -------------------------------------------------
test_scores = {}  # batch_size -> test CompWA
dataset_type = experiment_data.get("dataset_type", "SPR_synth")

for key, subdict in experiment_data.get("batch_size", {}).items():
    try:
        bs = key.split("_")[-1]
        epochs = subdict["epochs"]
        tr_loss = subdict["losses"]["train"]
        va_loss = subdict["losses"]["val"]
        va_comp = subdict["metrics"]["val_compwa"]
        test_scores[int(bs)] = subdict["metrics"]["test_compwa"]

        # ---- Loss curve ------------------------------------------------------
        try:
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, va_loss, label="Validation")
            plt.title(f"Loss Curve (Dataset: {dataset_type}, BS={bs})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = f"loss_curve_{dataset_type}_bs{bs}.png"
            plt.savefig(os.path.join(working_dir, fname))
        except Exception as e:
            print(f"Error creating loss plot for bs={bs}: {e}")
        finally:
            plt.close()

        # ---- CompWA curve ----------------------------------------------------
        try:
            plt.figure()
            plt.plot(epochs, va_comp)
            plt.title(f"Validation CompWA (Dataset: {dataset_type}, BS={bs})")
            plt.xlabel("Epoch")
            plt.ylabel("CompWA")
            fname = f"compwa_curve_{dataset_type}_bs{bs}.png"
            plt.savefig(os.path.join(working_dir, fname))
        except Exception as e:
            print(f"Error creating CompWA plot for bs={bs}: {e}")
        finally:
            plt.close()

    except Exception as e:
        print(f"Unexpected error while plotting for {key}: {e}")

# ---------- summary bar plot (max 1 figure) ----------------------------------
try:
    if test_scores:
        plt.figure()
        bs_vals = list(test_scores.keys())
        scores = [test_scores[bs] for bs in bs_vals]
        plt.bar([str(b) for b in bs_vals], scores)
        plt.title(f"Test CompWA by Batch Size (Dataset: {dataset_type})")
        plt.xlabel("Batch Size")
        plt.ylabel("Test CompWA")
        fname = f"test_compwa_summary_{dataset_type}.png"
        plt.savefig(os.path.join(working_dir, fname))
    else:
        print("No test CompWA data found to plot.")
except Exception as e:
    print(f"Error creating summary bar plot: {e}")
finally:
    plt.close()

# ---------- print evaluation metrics -----------------------------------------
if test_scores:
    print("\n=== Test CompWA by batch size ===")
    for bs, sc in sorted(test_scores.items()):
        print(f"  bs={bs:>3}: {sc:.4f}")
    best_bs = max(test_scores, key=test_scores.get)
    print(f"\nBest batch size: {best_bs}  (CompWA={test_scores[best_bs]:.4f})")
