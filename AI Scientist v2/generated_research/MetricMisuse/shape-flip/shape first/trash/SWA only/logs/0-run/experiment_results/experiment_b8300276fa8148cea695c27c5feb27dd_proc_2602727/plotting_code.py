import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_runs = experiment_data["dropout_rate"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_runs = {}

if not spr_runs:
    print("No SPR_BENCH data found; exiting.")
else:
    # --------- collect arrays ----------
    dropouts = sorted(float(k) for k in spr_runs.keys())
    val_curves = {d: spr_runs[str(d)]["metrics"]["val"] for d in dropouts}
    train_losses = {d: spr_runs[str(d)]["losses"]["train"] for d in dropouts}
    test_accs = [spr_runs[str(d)]["metrics"]["test"]["acc"] for d in dropouts]
    swa_vals = [spr_runs[str(d)]["metrics"]["test"]["swa"] for d in dropouts]
    cwa_vals = [spr_runs[str(d)]["metrics"]["test"]["cwa"] for d in dropouts]
    nrg_vals = [spr_runs[str(d)]["NRGS"] for d in dropouts]

    # -------------- plotting -------------
    # 1) Validation accuracy curves
    try:
        plt.figure()
        for d in dropouts:
            plt.plot(val_curves[d], label=f"drop={d}")
        plt.title("SPR_BENCH Validation Accuracy vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "spr_val_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating val accuracy plot: {e}")
        plt.close()

    # 2) Test accuracy by dropout
    try:
        plt.figure()
        plt.bar([str(d) for d in dropouts], test_accs, color="skyblue")
        plt.title("SPR_BENCH Test Accuracy by Dropout Rate")
        plt.xlabel("Dropout Rate")
        plt.ylabel("Accuracy")
        fname = os.path.join(working_dir, "spr_test_accuracy_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy plot: {e}")
        plt.close()

    # 3) SWA by dropout
    try:
        plt.figure()
        plt.plot(dropouts, swa_vals, marker="o")
        plt.title("SPR_BENCH Shape-Weighted Accuracy (SWA) by Dropout")
        plt.xlabel("Dropout Rate")
        plt.ylabel("SWA")
        fname = os.path.join(working_dir, "spr_swa_plot.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # 4) CWA by dropout
    try:
        plt.figure()
        plt.plot(dropouts, cwa_vals, marker="s", color="orange")
        plt.title("SPR_BENCH Color-Weighted Accuracy (CWA) by Dropout")
        plt.xlabel("Dropout Rate")
        plt.ylabel("CWA")
        fname = os.path.join(working_dir, "spr_cwa_plot.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CWA plot: {e}")
        plt.close()

    # 5) NRGS by dropout
    try:
        plt.figure()
        plt.bar([str(d) for d in dropouts], nrg_vals, color="green")
        plt.title("SPR_BENCH NRGS Score by Dropout")
        plt.xlabel("Dropout Rate")
        plt.ylabel("NRGS")
        fname = os.path.join(working_dir, "spr_nrg_score_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating NRGS plot: {e}")
        plt.close()

    # -------------- print summary ----------------
    print("Dropout | TestAcc | SWA | CWA | NRGS")
    for d, a, s, c, n in zip(dropouts, test_accs, swa_vals, cwa_vals, nrg_vals):
        print(f"{d:7.2f} | {a:7.3f} | {s:.3f} | {c:.3f} | {n:.3f}")
