import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

wd_dict = experiment_data.get("weight_decay", {})
tags = list(wd_dict.keys())


# helper to get arrays
def arr(tag, field, sub):
    return np.asarray(wd_dict[tag][field][sub])


# 1) TRAIN LOSSES ---------------------------------------------------
try:
    plt.figure()
    for tag in tags:
        plt.plot(arr(tag, "losses", "train"), label=f"wd={tag}")
    plt.title("SPR: Train Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_train_loss_vs_epoch.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating train-loss plot: {e}")
    plt.close()

# 2) VAL LOSSES -----------------------------------------------------
try:
    plt.figure()
    for tag in tags:
        plt.plot(arr(tag, "losses", "val"), label=f"wd={tag}")
    plt.title("SPR: Validation Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_val_loss_vs_epoch.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val-loss plot: {e}")
    plt.close()

# 3) SCHM over epochs ----------------------------------------------
try:
    plt.figure()
    for tag in tags:
        plt.plot(arr(tag, "metrics", "SCHM"), label=f"wd={tag}")
    plt.title("SPR: SCHM Metric vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("SCHM")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_SCHM_vs_epoch.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SCHM-epoch plot: {e}")
    plt.close()

# 4) Final SCHM bar -------------------------------------------------
try:
    plt.figure()
    final_schm = [arr(t, "metrics", "SCHM")[-1] for t in tags]
    plt.bar(tags, final_schm, color="skyblue")
    plt.title("SPR: Final SCHM by Weight Decay")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final SCHM")
    fname = os.path.join(working_dir, "SPR_final_SCHM_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final SCHM bar: {e}")
    plt.close()

# 5) SWA vs CWA scatter --------------------------------------------
try:
    plt.figure()
    swa = [arr(t, "metrics", "SWA")[-1] for t in tags]
    cwa = [arr(t, "metrics", "CWA")[-1] for t in tags]
    plt.scatter(swa, cwa)
    for s, c, t in zip(swa, cwa, tags):
        plt.text(s, c, t)
    plt.title("SPR: Final SWA vs CWA (Weight Decay tags)")
    plt.xlabel("SWA")
    plt.ylabel("CWA")
    fname = os.path.join(working_dir, "SPR_SWA_vs_CWA_scatter.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA-CWA scatter: {e}")
    plt.close()

# --------- quick textual summary -------------
for tag in tags:
    print(f"wd={tag} | Final SCHM={arr(tag,'metrics','SCHM')[-1]:.3f}")
