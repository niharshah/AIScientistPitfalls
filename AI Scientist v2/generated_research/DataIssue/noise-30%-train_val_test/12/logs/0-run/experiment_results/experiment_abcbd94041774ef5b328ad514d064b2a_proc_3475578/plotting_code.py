import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ LOAD DATA ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

summary = []  # (batch, last_val_f1)

# ------------------ PLOTS ----------------------
for exp_name, dsets in experiment_data.items():
    for dset_name, cfg in dsets.items():
        for bs, stats in cfg.get("batch_size", {}).items():
            epochs = stats["epochs"]
            tr_loss = stats["losses"]["train"]
            val_loss = stats["losses"]["val"]
            val_f1 = stats["metrics"]["val"]
            last_f1 = val_f1[-1] if val_f1 else float("nan")
            summary.append((bs, last_f1))

            # Loss curves
            try:
                plt.figure()
                plt.plot(epochs, tr_loss, label="Train Loss")
                plt.plot(epochs, val_loss, label="Val Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{dset_name} Loss Curves (bs={bs})")
                plt.legend()
                fname = f"{dset_name}_loss_bs{bs}.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating loss plot for bs{bs}: {e}")
                plt.close()

            # F1 curves
            try:
                plt.figure()
                plt.plot(epochs, val_f1, label="Val F1")
                plt.xlabel("Epoch")
                plt.ylabel("Macro F1")
                plt.title(f"{dset_name} Validation F1 (bs={bs})")
                plt.legend()
                fname = f"{dset_name}_f1_bs{bs}.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating F1 plot for bs{bs}: {e}")
                plt.close()

# ------------- SUMMARY BAR CHART --------------
try:
    if summary:
        bss, f1s = zip(*sorted(summary))
        plt.figure()
        plt.bar(range(len(bss)), f1s, tick_label=bss)
        plt.xlabel("Batch Size")
        plt.ylabel("Final Val Macro F1")
        plt.title("SPR-BENCH Final Validation F1 vs Batch Size")
        plt.savefig(os.path.join(working_dir, "SPR-BENCH_final_f1_bar.png"))
        plt.close()
except Exception as e:
    print(f"Error creating summary bar chart: {e}")
    plt.close()

# ------------- PRINT SUMMARY ------------------
print("Batch Size | Final Val F1")
for bs, f1 in sorted(summary):
    print(f"{bs:10} | {f1:.3f}")
