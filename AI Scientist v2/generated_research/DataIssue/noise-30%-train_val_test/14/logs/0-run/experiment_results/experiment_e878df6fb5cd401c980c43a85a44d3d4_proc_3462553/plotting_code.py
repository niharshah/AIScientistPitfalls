import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    wd_dict = experiment_data.get("weight_decay", {})
    best_f1, best_wd = -1.0, None

    # Iterate through weight_decay runs (max 5)
    for idx, (wd, run) in enumerate(sorted(wd_dict.items(), key=lambda x: float(x[0]))):
        tr_loss = [pt["loss"] for pt in run["losses"]["train"]]
        val_loss = [pt["loss"] for pt in run["losses"]["val"]]
        tr_f1 = [pt["macro_f1"] for pt in run["metrics"]["train"]]
        val_f1 = [pt["macro_f1"] for pt in run["metrics"]["val"]]
        epochs = list(range(1, len(tr_loss) + 1))

        # Track best
        if val_f1[-1] > best_f1:
            best_f1, best_wd = val_f1[-1], wd

        # ----------------------------------------------------------
        try:
            plt.figure(figsize=(10, 4))

            # Left subplot: Loss
            plt.subplot(1, 2, 1)
            plt.plot(epochs, tr_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss")

            # Right subplot: Macro F1
            plt.subplot(1, 2, 2)
            plt.plot(epochs, tr_f1, label="Train F1")
            plt.plot(epochs, val_f1, label="Val F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro F1")
            plt.title("Macro F1")

            plt.suptitle(f"SPR – weight_decay={wd}\nLeft: Loss, Right: Macro F1")
            plt.legend()

            fname = f"spr_wd_{wd.replace('.', 'p')}_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating plot for wd={wd}: {e}")
            plt.close()

    # --------------------------------------------------------------
    print("\nFinal validation Macro-F1 by weight_decay:")
    for wd, run in sorted(wd_dict.items(), key=lambda x: float(x[0])):
        final_f1 = run["metrics"]["val"][-1]["macro_f1"]
        print(f"  wd={wd}: {final_f1:.4f}")
    print(f"\nBest weight_decay = {best_wd}  (Val Macro-F1 = {best_f1:.4f})")
