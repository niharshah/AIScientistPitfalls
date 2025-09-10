import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment log -------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit(0)

# ---------------- iterate through datasets ------------------------
for dset_name, dset_entry in experiment_data.items():
    losses = dset_entry.get("losses", {})
    metrics = dset_entry.get("metrics", {})
    meta = dset_entry.get("meta", {})
    if not losses:
        print(f"No loss data for {dset_name}; skipping.")
        continue

    # ---------- gather arrays -------------
    tr_loss = np.asarray(losses.get("train", []))
    val_loss = np.asarray(losses.get("val", []))
    swa_val = np.asarray(metrics.get("val", []))
    epochs = np.arange(1, len(tr_loss) + 1)

    # ---------- plot 1: loss curves -------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, "--o", label="train")
        plt.plot(epochs, val_loss, "-o", label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset_name}: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset_name}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset_name}: {e}")
        plt.close()

    # ---------- plot 2: validation SWA ----
    try:
        if swa_val.size:
            plt.figure()
            plt.plot(epochs, swa_val, "-o")
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.title(f"{dset_name}: Validation Shape-Weighted Accuracy")
            fname = os.path.join(working_dir, f"{dset_name}_val_SWA.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {dset_name}: {e}")
        plt.close()

    # ---------- plot 3: test SWA bar ------
    try:
        # collect keys that look like SWA_test_* in meta
        variant_names, swa_tests = [], []
        for k, v in meta.items():
            if k.startswith("SWA_test_"):
                variant_names.append(k.replace("SWA_test_", ""))
                swa_tests.append(v)
        if swa_tests:
            plt.figure()
            x = np.arange(len(variant_names))
            plt.bar(x, swa_tests, color="skyblue")
            plt.xticks(x, variant_names)
            plt.ylabel("SWA")
            plt.title(f"{dset_name}: Final Test SWA by Variant")
            fname = os.path.join(working_dir, f"{dset_name}_test_SWA_bar.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating test SWA bar for {dset_name}: {e}")
        plt.close()

    # ---------- print summary -------------
    if swa_tests:
        print(f"\n{dset_name} – Final Test SWA:")
        for n, v in zip(variant_names, swa_tests):
            print(f"  {n:12s}: {v:.4f}")
