import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment results ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    runs = experiment_data.get("num_gcn_layers", {})
    # ---- per-depth loss & accuracy curves ----
    for depth_str, run_dict in runs.items():
        try:
            ed = run_dict["SPR_BENCH"]
            tr_loss, va_loss = ed["losses"]["train"], ed["losses"]["val"]
            tr_acc, va_acc = ed["metrics"]["train"], ed["metrics"]["val"]
            epochs = range(1, len(tr_loss) + 1)

            plt.figure(figsize=(10, 4))
            # Left subplot: loss
            plt.subplot(1, 2, 1)
            plt.plot(epochs, tr_loss, label="train")
            plt.plot(epochs, va_loss, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Loss")

            # Right subplot: accuracy
            plt.subplot(1, 2, 2)
            plt.plot(epochs, tr_acc, label="train")
            plt.plot(epochs, va_acc, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.ylim(0, 1)
            plt.legend()
            plt.title("Accuracy")

            plt.suptitle(
                f"SPR_BENCH – {depth_str} GCN layers\nLeft: Loss, Right: Accuracy"
            )
            fname = f"SPR_BENCH_depth_{depth_str}_loss_acc.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating plot for depth {depth_str}: {e}")
            plt.close()

    # ---- bar chart of complexity-weighted accuracy ----
    try:
        depths, compwa_vals = [], []
        for depth_str, run_dict in runs.items():
            depths.append(int(depth_str))
            compwa_vals.append(run_dict["SPR_BENCH"].get("compWA", 0.0))
        plt.figure()
        plt.bar([str(d) for d in depths], compwa_vals, color="skyblue")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title("SPR_BENCH – Complexity-Weighted Accuracy by GCN Depth")
        fname = "SPR_BENCH_compWA_bar.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating compWA bar chart: {e}")
        plt.close()
