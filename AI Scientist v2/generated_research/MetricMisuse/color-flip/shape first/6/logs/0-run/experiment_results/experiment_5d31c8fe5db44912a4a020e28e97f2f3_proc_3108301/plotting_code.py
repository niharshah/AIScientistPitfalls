import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

colors = plt.cm.tab10.colors

for dsi, (ds_name, ds_dict) in enumerate(experiment_data.items()):
    # ---- helper to safely fetch arrays ----
    def arr(*keys):
        try:
            return np.asarray(ds_dict[keys[0]][keys[1]])
        except Exception:
            return None

    tr_loss = arr("losses", "train")
    val_loss = arr("losses", "val")
    swa = arr("metrics", "train")  # might hold SWA if author stored it here
    cwa = arr("metrics", "val")  # might hold CWA if author stored it here
    scaa = arr("SCAA", "val")

    epochs = None
    if val_loss is not None:
        epochs = np.arange(1, len(val_loss) + 1)

    # -------- PLOT 1 : loss curves ---------
    try:
        if tr_loss is not None and val_loss is not None:
            plt.figure()
            plt.plot(epochs, tr_loss, linestyle="--", color=colors[0], label="train")
            plt.plot(epochs, val_loss, linestyle="-", color=colors[1], label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"Loss Curves – Dataset: {ds_name}\nLeft: Train (--), Right: Val (—)"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Error plotting loss for {ds_name}: {e}")
        plt.close()

    # -------- PLOT 2 : SWA curve -----------
    try:
        if swa is not None:
            plt.figure()
            plt.plot(np.arange(1, len(swa) + 1), swa, color=colors[2])
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title(f"SWA Across Epochs – Dataset: {ds_name}")
            fname = os.path.join(working_dir, f"{ds_name}_SWA_curve.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Error plotting SWA for {ds_name}: {e}")
        plt.close()

    # -------- PLOT 3 : CWA curve -----------
    try:
        if cwa is not None:
            plt.figure()
            plt.plot(np.arange(1, len(cwa) + 1), cwa, color=colors[3])
            plt.xlabel("Epoch")
            plt.ylabel("Color-Weighted Accuracy")
            plt.title(f"CWA Across Epochs – Dataset: {ds_name}")
            fname = os.path.join(working_dir, f"{ds_name}_CWA_curve.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Error plotting CWA for {ds_name}: {e}")
        plt.close()

    # -------- PLOT 4 : SCAA curve ----------
    try:
        if scaa is not None:
            plt.figure()
            plt.plot(np.arange(1, len(scaa) + 1), scaa, color=colors[4])
            plt.xlabel("Epoch")
            plt.ylabel("SCAA")
            plt.title(f"SCAA Across Epochs – Dataset: {ds_name}")
            fname = os.path.join(working_dir, f"{ds_name}_SCAA_curve.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Error plotting SCAA for {ds_name}: {e}")
        plt.close()

# -------- optional: print final metrics -----
for ds_name, ds_dict in experiment_data.items():
    try:
        best_scaa = max(ds_dict["SCAA"]["val"])
        print(f"{ds_name}: best Val SCAA = {best_scaa:.3f}")
    except Exception:
        pass
