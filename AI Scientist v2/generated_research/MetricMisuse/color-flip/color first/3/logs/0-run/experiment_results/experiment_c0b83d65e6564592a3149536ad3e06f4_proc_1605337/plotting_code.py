import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------- Load experiment data --------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

final_summary = []

# --------------------- Produce up to 5 figures ------------------ #
if experiment_data:
    spr_data = experiment_data["hidden_dim"]["SPR_BENCH"]
    for i, (hd, edict) in enumerate(sorted(spr_data.items())):
        if i >= 5:  # safety guard, though we only have 5 dims
            break
        try:
            epochs = [ep for ep, _ in edict["losses"]["train"]]
            train_loss = [l for _, l in edict["losses"]["train"]]
            val_loss = [l for _, l in edict["losses"]["val"]]
            val_metrics = np.array(edict["metrics"]["val"])  # (epoch,CWA,SWA,HCSA)
            cwa = val_metrics[:, 1]
            swa = val_metrics[:, 2]
            hcs = val_metrics[:, 3]

            plt.figure(figsize=(10, 4))

            # Left subplot: losses
            plt.subplot(1, 2, 1)
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Curves")
            plt.legend()

            # Right subplot: metrics
            plt.subplot(1, 2, 2)
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, hcs, label="HCSA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("Validation Metrics")
            plt.legend()

            plt.suptitle(
                f"SPR_BENCH HiddenDim={hd}\nLeft: Losses, Right: Validation Metrics"
            )
            fname = f"SPR_BENCH_hidden{hd}_train_val_curves.png"
            plt.tight_layout(rect=[0, 0, 1, 0.90])
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating plot for hidden_dim={hd}: {e}")
            plt.close()

        # collect final dev/test HCSA for quick text summary
        dev_hcs = edict["metrics"]["dev"][2]
        test_hcs = edict["metrics"]["test"][2]
        final_summary.append((hd, dev_hcs, test_hcs))

# --------------------- Print final summary ---------------------- #
if final_summary:
    print("\nHiddenDim | Dev HCSA | Test HCSA")
    for hd, d, t in sorted(final_summary):
        print(f"{hd:9d} | {d:.3f}    | {t:.3f}")
