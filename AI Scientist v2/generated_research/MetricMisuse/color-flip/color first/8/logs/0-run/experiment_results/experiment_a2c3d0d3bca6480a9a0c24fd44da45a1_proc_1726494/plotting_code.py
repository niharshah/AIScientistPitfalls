import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------
# plotting
created = []
try:
    act_dict = experiment_data["activation_function"]["spr_bench"]
except Exception as e:
    print(f"Could not locate spr_bench data: {e}")
    act_dict = {}

max_figs = 5
for i, (act_name, log) in enumerate(act_dict.items()):
    if i >= max_figs:
        break
    try:
        # -------- collect data --------
        ep_tr, loss_tr = zip(*log["losses"]["train"])
        _, loss_va = zip(*log["losses"]["val"])
        _, cwa_tr, swa_tr, cshm_tr = zip(*log["metrics"]["train"])
        _, cwa_va, swa_va, cshm_va = zip(*log["metrics"]["val"])

        # -------- plot --------
        plt.figure(figsize=(10, 4))
        plt.suptitle(f"SPR_BENCH – {act_name.upper()}")

        # left subplot: Loss
        plt.subplot(1, 2, 1)
        plt.plot(ep_tr, loss_tr, label="Train")
        plt.plot(ep_tr, loss_va, label="Validation")
        plt.title("Left: Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.grid(True)

        # right subplot: CSHM metric
        plt.subplot(1, 2, 2)
        plt.plot(ep_tr, cshm_tr, label="Train")
        plt.plot(ep_tr, cshm_va, label="Validation")
        plt.title("Right: CSHM Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Mean (CWA,SWA)")
        plt.legend()
        plt.grid(True)

        # save & close
        fname = f"spr_bench_{act_name}_loss_cshm.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        created.append(fname)
    except Exception as e:
        print(f"Error creating plot for {act_name}: {e}")
        plt.close()

print("Created figures:", ", ".join(created) if created else "none")
