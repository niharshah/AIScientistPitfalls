import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data_root = experiment_data.get("nhead_tuning", {}).get("SPR_BENCH", {})


# ------------------ helper ------------------
def extract_curve(nhead_str, key_outer, key_inner, val_key):
    recs = data_root[nhead_str][key_outer][key_inner]
    return [d["epoch"] for d in recs], [d[val_key] for d in recs]


# ------------------ figure 1: loss curves ------------------
try:
    plt.figure()
    for nhead_str in sorted(data_root.keys(), key=lambda x: int(x)):
        ep_tr, tr_loss = extract_curve(nhead_str, "losses", "train", "loss")
        _, val_loss = extract_curve(nhead_str, "losses", "val", "loss")
        plt.plot(ep_tr, tr_loss, label=f"train nhead={nhead_str}")
        plt.plot(ep_tr, val_loss, linestyle="--", label=f"val nhead={nhead_str}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nTrain vs Validation for different nhead values")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_nhead.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves figure: {e}")
    plt.close()

# ------------------ figure 2: macro-F1 curves ------------------
try:
    plt.figure()
    for nhead_str in sorted(data_root.keys(), key=lambda x: int(x)):
        ep_tr, tr_f1 = extract_curve(nhead_str, "metrics", "train", "macro_f1")
        _, val_f1 = extract_curve(nhead_str, "metrics", "val", "macro_f1")
        plt.plot(ep_tr, tr_f1, label=f"train nhead={nhead_str}")
        plt.plot(ep_tr, val_f1, linestyle="--", label=f"val nhead={nhead_str}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(
        "SPR_BENCH Macro-F1 Curves\nTrain vs Validation for different nhead values"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curves_nhead.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating macro-F1 curves figure: {e}")
    plt.close()

# ------------------ figure 3: final val F1 bar chart ------------------
best_scores = {}
try:
    for nhead_str in data_root.keys():
        _, val_f1 = extract_curve(nhead_str, "metrics", "val", "macro_f1")
        best_scores[int(nhead_str)] = max(val_f1)
    plt.figure()
    heads = sorted(best_scores.keys())
    scores = [best_scores[h] for h in heads]
    plt.bar(heads, scores, color="skyblue")
    plt.xlabel("nhead")
    plt.ylabel("Best Val Macro-F1")
    plt.title("SPR_BENCH Final Validation Macro-F1 vs nhead")
    fname = os.path.join(working_dir, "SPR_BENCH_final_valF1_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final val F1 bar chart: {e}")
    plt.close()

# ------------------ print evaluation summary ------------------
if best_scores:
    print("Best Validation Macro-F1 per nhead:")
    for h, sc in sorted(best_scores.items()):
        print(f"  nhead={h}: {sc:.4f}")
