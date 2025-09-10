import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load --------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    bs_dict = experiment_data.get("batch_size", {}).get("SPR_BENCH", {})
    batch_sizes = sorted(bs_dict.keys())

    # --------- 1) loss curves per batch size ------------
    for bs in batch_sizes:
        try:
            tr_loss = bs_dict[bs]["losses"]["train"]
            vl_loss = bs_dict[bs]["losses"]["val"]
            plt.figure()
            plt.plot(tr_loss, label="Train")
            plt.plot(vl_loss, label="Validation")
            plt.title(f"SPR_BENCH Loss vs Epoch (bs={bs})")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            fname = f"SPR_BENCH_loss_curve_bs{bs}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for bs={bs}: {e}")
            plt.close()

    # --------- 2) combined CWA curves -------------------
    try:
        plt.figure()
        for bs in batch_sizes:
            cwa = bs_dict[bs]["metrics"]["val"]
            plt.plot(range(1, len(cwa) + 1), cwa, label=f"bs={bs}")
        plt.title("SPR_BENCH Validation CWA vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.legend()
        fname = "SPR_BENCH_CWA_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating combined CWA plot: {e}")
        plt.close()

    # --------- 3) final CWA bar chart -------------------
    try:
        final_cwa = [bs_dict[bs]["metrics"]["val"][-1] for bs in batch_sizes]
        plt.figure()
        plt.bar([str(bs) for bs in batch_sizes], final_cwa, color="skyblue")
        plt.title("SPR_BENCH Final Validation CWA by Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("Final CWA")
        fname = "SPR_BENCH_final_CWA_bar.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating final CWA bar plot: {e}")
        plt.close()
