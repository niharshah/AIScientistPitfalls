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
    runs = experiment_data["DROPOUT_PROB"]["SPR_BENCH"]["dropouts"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

dropouts = sorted([float(k) for k in runs.keys()])
colors = plt.cm.viridis(np.linspace(0, 1, len(dropouts)))

# ---------- find best dropout ----------
best_dp, best_val_mcc = None, -1
for d in dropouts:
    val_mcc = runs[str(d)]["metrics"]["val_MCC"][-1]
    if val_mcc > best_val_mcc:
        best_val_mcc, best_dp = val_mcc, d
print(f"Best dropout = {best_dp} with final val_MCC = {best_val_mcc:.4f}")
print(f'Test MCC at best dropout = {runs[str(best_dp)]["metrics"]["test_MCC"]:.4f}')

# ---------- plot 1: combined val loss ----------
try:
    plt.figure()
    for d, c in zip(dropouts, colors):
        ep = runs[str(d)]["epochs"]
        plt.plot(ep, runs[str(d)]["losses"]["val"], label=f"dropout={d}", color=c)
    plt.title("Validation Loss vs Epochs (SPR_BENCH)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fn = os.path.join(working_dir, "spr_bench_val_loss_all_dropouts.png")
    plt.savefig(fn)
    plt.close()
except Exception as e:
    print(f"Error creating combined val loss plot: {e}")
    plt.close()

# ---------- plot 2: combined val MCC ----------
try:
    plt.figure()
    for d, c in zip(dropouts, colors):
        ep = runs[str(d)]["epochs"]
        plt.plot(ep, runs[str(d)]["metrics"]["val_MCC"], label=f"dropout={d}", color=c)
    plt.title("Validation MCC vs Epochs (SPR_BENCH)")
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.legend()
    fn = os.path.join(working_dir, "spr_bench_val_mcc_all_dropouts.png")
    plt.savefig(fn)
    plt.close()
except Exception as e:
    print(f"Error creating combined val MCC plot: {e}")
    plt.close()

# ---------- plot 3: bar of test MCC ----------
try:
    plt.figure()
    test_vals = [runs[str(d)]["metrics"]["test_MCC"] for d in dropouts]
    plt.bar([str(d) for d in dropouts], test_vals, color=colors)
    plt.title("Test MCC by Dropout (SPR_BENCH)")
    plt.xlabel("Dropout Probability")
    plt.ylabel("Test MCC")
    fn = os.path.join(working_dir, "spr_bench_test_mcc_bar.png")
    plt.savefig(fn)
    plt.close()
except Exception as e:
    print(f"Error creating test MCC bar plot: {e}")
    plt.close()

# ---------- plot 4: best run loss curves ----------
try:
    rec = runs[str(best_dp)]
    plt.figure()
    plt.plot(rec["epochs"], rec["losses"]["train"], label="train_loss")
    plt.plot(rec["epochs"], rec["losses"]["val"], label="val_loss")
    plt.title(f"Loss Curves (Best Dropout={best_dp})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fn = os.path.join(working_dir, f"spr_bench_best_dropout_{best_dp}_loss.png")
    plt.savefig(fn)
    plt.close()
except Exception as e:
    print(f"Error creating best loss curve: {e}")
    plt.close()

# ---------- plot 5: best run MCC curves ----------
try:
    rec = runs[str(best_dp)]
    plt.figure()
    plt.plot(rec["epochs"], rec["metrics"]["train_MCC"], label="train_MCC")
    plt.plot(rec["epochs"], rec["metrics"]["val_MCC"], label="val_MCC")
    plt.title(f"MCC Curves (Best Dropout={best_dp})")
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.legend()
    fn = os.path.join(working_dir, f"spr_bench_best_dropout_{best_dp}_mcc.png")
    plt.savefig(fn)
    plt.close()
except Exception as e:
    print(f"Error creating best MCC curve: {e}")
    plt.close()
