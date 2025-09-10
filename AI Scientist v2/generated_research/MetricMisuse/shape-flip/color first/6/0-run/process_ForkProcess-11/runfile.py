import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up working dir ------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ---------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- per-dataset plots -------------------------------------------------
test_cplx_summary = {}  # dataset -> test CplxWA

for ds_name, ed in experiment_data.items():
    if not isinstance(ed, dict):
        continue  # skip accidental non-dataset keys
    epochs = ed.get("epochs", [])
    if not epochs:
        continue

    # ---------- Loss curve ----------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, ed["losses"]["train"], label="Train")
        plt.plot(epochs, ed["losses"]["val"], label="Validation")
        plt.title(f"{ds_name}: Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = f"loss_curve_{ds_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
    finally:
        plt.close()

    # ---------- Validation CWA ------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["val"]["CWA"])
        plt.title(f"{ds_name}: Validation CWA")
        plt.xlabel("Epoch")
        plt.ylabel("CWA")
        fname = f"val_cwa_{ds_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error creating CWA plot for {ds_name}: {e}")
    finally:
        plt.close()

    # ---------- Validation SWA ------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["val"]["SWA"])
        plt.title(f"{ds_name}: Validation SWA")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        fname = f"val_swa_{ds_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error creating SWA plot for {ds_name}: {e}")
    finally:
        plt.close()

    # ---------- Validation CplxWA --------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, ed["metrics"]["val"]["CplxWA"])
        plt.title(f"{ds_name}: Validation CplxWA")
        plt.xlabel("Epoch")
        plt.ylabel("CplxWA")
        fname = f"val_cplxwa_{ds_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error creating CplxWA plot for {ds_name}: {e}")
    finally:
        plt.close()

    # ---------- collect test metric ------------------------------------------
    tst = ed.get("metrics", {}).get("test", {})
    if "CplxWA" in tst:
        test_cplx_summary[ds_name] = tst["CplxWA"]

    # ---------- print test metrics -------------------------------------------
    if tst:
        print(
            f"{ds_name}  Test CWA={tst.get('CWA', np.nan):.3f}  "
            f"SWA={tst.get('SWA', np.nan):.3f}  "
            f"CplxWA={tst.get('CplxWA', np.nan):.3f}"
        )

# ---------- summary bar plot --------------------------------------------------
try:
    if test_cplx_summary:
        plt.figure()
        names = list(test_cplx_summary.keys())
        scores = [test_cplx_summary[n] for n in names]
        plt.bar(names, scores)
        plt.title("Test CplxWA by Dataset")
        plt.xlabel("Dataset")
        plt.ylabel("Test CplxWA")
        fname = "summary_test_cplxwa.png"
        plt.savefig(os.path.join(working_dir, fname))
    else:
        print("No test CplxWA data found for summary plot.")
except Exception as e:
    print(f"Error creating summary bar plot: {e}")
finally:
    plt.close()
