import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
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

# ---------- plotting ----------
spr_dict = experiment_data.get("hidden_dim", {}).get("SPR_BENCH", {})
hidden_dims = list(spr_dict.keys())[:5]  # ensure at most 5 curves if more exist

# 1-4: accuracy curves per hidden dim
for hid in hidden_dims:
    try:
        met = spr_dict[hid]["metrics"]
        acc_tr, acc_val = met["train"], met["val"]
        plt.figure()
        plt.plot(acc_tr, label="Train")
        plt.plot(acc_val, label="Validation")
        plt.title(f"SPR_BENCH – Train vs Validation Accuracy (hid={hid})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = f"SPR_BENCH_acc_curve_hid{hid}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve for hid={hid}: {e}")
        plt.close()

# 5: bar chart of Comp-Weighted Accuracy across hidden dims
try:
    comp_accs = [spr_dict[h]["comp_weighted_acc"] for h in hidden_dims]
    plt.figure()
    plt.bar(range(len(hidden_dims)), comp_accs, tick_label=hidden_dims)
    plt.title("SPR_BENCH – Comp-Weighted Accuracy by Hidden Dimension")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Comp-Weighted Accuracy")
    fname = "SPR_BENCH_comp_weighted_accuracy.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
    # print metrics
    print("Comp-Weighted Accuracies:")
    for h, a in zip(hidden_dims, comp_accs):
        print(f"  hid={h}: {a:.4f}")
except Exception as e:
    print(f"Error creating Comp-Weighted Accuracy bar chart: {e}")
    plt.close()
