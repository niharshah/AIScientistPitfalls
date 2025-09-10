import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    tags = list(experiment_data["pretrain_epochs"].keys())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    tags = []

# collect final metrics
final_swa, final_cwa, final_schm = {}, {}, {}
for t in tags:
    m = experiment_data["pretrain_epochs"][t]["metrics"]
    final_swa[t] = m["SWA"][-1]
    final_cwa[t] = m["CWA"][-1]
    final_schm[t] = m["SCHM"][-1]

# ------------- 1. loss curves -----------------
try:
    plt.figure()
    for t in tags:
        train = experiment_data["pretrain_epochs"][t]["losses"]["train"]
        val = experiment_data["pretrain_epochs"][t]["losses"]["val"]
        plt.plot(train, label=f"{t}-train")
        plt.plot(val, linestyle="--", label=f"{t}-val")
    plt.title("SPR_BENCH: Fine-tune Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------- 2. SCHM curves -----------------
try:
    plt.figure()
    for t in tags:
        schm = experiment_data["pretrain_epochs"][t]["metrics"]["SCHM"]
        plt.plot(schm, label=t)
    plt.title("SPR_BENCH: SCHM over Fine-tuning Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SCHM")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_schm_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SCHM curves: {e}")
    plt.close()

# ------------- 3. final SCHM bar --------------
try:
    plt.figure()
    plt.bar(range(len(tags)), [final_schm[t] for t in tags], tick_label=tags)
    plt.title("SPR_BENCH: Final SCHM by Pre-train Epochs")
    plt.ylabel("SCHM")
    fname = os.path.join(working_dir, "spr_bench_final_schm_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SCHM bar: {e}")
    plt.close()

# ------------- 4. final SWA & CWA -------------
try:
    x = np.arange(len(tags))
    width = 0.35
    plt.figure()
    plt.bar(x - width / 2, [final_swa[t] for t in tags], width, label="SWA")
    plt.bar(x + width / 2, [final_cwa[t] for t in tags], width, label="CWA")
    plt.xticks(x, tags)
    plt.title("SPR_BENCH: Final SWA vs CWA")
    plt.ylabel("Weighted Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_final_swa_cwa_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA/CWA bar: {e}")
    plt.close()

# ------------- 5. SWA vs CWA scatter ----------
try:
    plt.figure()
    for t in tags:
        plt.scatter(final_swa[t], final_cwa[t], label=t)
    plt.title("SPR_BENCH: Final SWA vs CWA")
    plt.xlabel("SWA")
    plt.ylabel("CWA")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_swa_vs_cwa_scatter.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA vs CWA scatter: {e}")
    plt.close()

# Print final metrics table
for t in tags:
    print(
        f"{t}: SWA={final_swa[t]:.3f}, CWA={final_cwa[t]:.3f}, SCHM={final_schm[t]:.3f}"
    )
