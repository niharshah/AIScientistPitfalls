import matplotlib.pyplot as plt
import numpy as np
import os

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

# pull out SPR_BENCH info
bench = experiment_data.get("SPR_BENCH", {})
train_loss = bench.get("losses", {}).get("train", [])
val_loss = bench.get("losses", {}).get("val", [])
swa = bench.get("metrics", {}).get("SWA", [])
cwa = bench.get("metrics", {}).get("CWA", [])
scwa = bench.get("metrics", {}).get("SCWA", [])
contrastive = bench.get("contrastive_loss", [])

sup_epochs = np.arange(1, len(train_loss) + 1)
con_epochs = np.arange(1, len(contrastive) + 1)

# 1) Loss curves ---------------------------------------------------------------
try:
    plt.figure()
    plt.plot(sup_epochs, train_loss, label="Train")
    plt.plot(sup_epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Metric curves -------------------------------------------------------------
try:
    plt.figure()
    plt.plot(sup_epochs, swa, label="SWA")
    plt.plot(sup_epochs, cwa, label="CWA")
    plt.plot(sup_epochs, scwa, label="SCWA")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("SPR_BENCH: Weighted Accuracy Metrics")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_metric_curves.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# 3) Contrastive pre-training loss --------------------------------------------
try:
    plt.figure()
    plt.plot(con_epochs, contrastive, marker="o")
    plt.xlabel("Pre-train Epoch")
    plt.ylabel("NT-Xent Loss")
    plt.title("SPR_BENCH: Contrastive Pre-training Loss")
    fname = os.path.join(working_dir, "spr_bench_contrastive_loss.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating contrastive plot: {e}")
    plt.close()

# 4) Final epoch metric snapshot ----------------------------------------------
try:
    final_vals = [swa[-1] if swa else 0, cwa[-1] if cwa else 0, scwa[-1] if scwa else 0]
    labels = ["SWA", "CWA", "SCWA"]
    plt.figure()
    plt.bar(labels, final_vals)
    plt.ylim(0, 1)
    plt.title("SPR_BENCH: Final Epoch Metrics")
    fname = os.path.join(working_dir, "spr_bench_final_metrics.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating final metric plot: {e}")
    plt.close()

# -------- print aggregate results --------------------------------------------
if train_loss and val_loss:
    print(f"Final Validation Loss: {val_loss[-1]:.4f}")
if swa and cwa and scwa:
    print(f"Final SWA={swa[-1]:.3f} | CWA={cwa[-1]:.3f} | SCWA={scwa[-1]:.3f}")
