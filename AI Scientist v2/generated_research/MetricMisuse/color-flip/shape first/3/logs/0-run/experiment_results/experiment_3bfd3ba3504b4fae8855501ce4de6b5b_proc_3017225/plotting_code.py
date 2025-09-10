import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load data ------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------ plotting ------------
for dataset, res in experiment_data.get("hid_dim", {}).items():
    # Gather hidden sizes sorted numerically
    hid_dims = sorted(res[dataset].keys(), key=int)
    # 1) Loss curves
    try:
        plt.figure(figsize=(8, 4))
        for hd in hid_dims:
            ep = res[dataset][hd]["epochs"]
            tr = res[dataset][hd]["losses"]["train"]
            vl = res[dataset][hd]["losses"]["val"]
            plt.plot(ep, tr, linestyle="--", label=f"train_loss hid={hd}")
            plt.plot(ep, vl, linestyle="-", label=f"val_loss hid={hd}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset}: Training/Validation Loss vs Epoch (various hid_dim)")
        plt.legend(fontsize=8)
        file_name = f"{dataset}_loss_curves.png"
        plt.savefig(os.path.join(working_dir, file_name))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2) Validation SCWA curves
    try:
        plt.figure(figsize=(6, 4))
        for hd in hid_dims:
            ep = res[dataset][hd]["epochs"]
            scwa = res[dataset][hd]["metrics"]["val"]
            plt.plot(ep, scwa, marker="o", label=f"hid={hd}")
        plt.xlabel("Epoch")
        plt.ylabel("SCWA")
        plt.title(f"{dataset}: Validation SCWA vs Epoch (various hid_dim)")
        plt.legend(fontsize=8)
        file_name = f"{dataset}_val_scwa_curves.png"
        plt.savefig(os.path.join(working_dir, file_name))
        plt.close()
    except Exception as e:
        print(f"Error creating SCWA plot: {e}")
        plt.close()

    # 3) Test SCWA bar chart
    try:
        plt.figure(figsize=(6, 4))
        test_scores = [res[dataset][hd]["test_SCWA"] for hd in hid_dims]
        plt.bar(range(len(hid_dims)), test_scores, tick_label=hid_dims)
        plt.xlabel("hid_dim")
        plt.ylabel("Test SCWA")
        plt.title(f"{dataset}: Test SCWA by Hidden Size")
        file_name = f"{dataset}_test_scwa_bar.png"
        plt.savefig(os.path.join(working_dir, file_name))
        plt.close()
    except Exception as e:
        print(f"Error creating Test SCWA bar plot: {e}")
        plt.close()

    # ------------ print metrics ------------
    best_idx = int(np.argmax(test_scores))
    print("\n=== Test SCWA Summary ===")
    for hd, sc in zip(hid_dims, test_scores):
        marker = "*" if hid_dims[best_idx] == hd else " "
        print(f"hid_dim={hd:>3}: Test SCWA={sc:.4f}{marker}")
