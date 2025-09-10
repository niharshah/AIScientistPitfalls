import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------- load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

bench = experiment_data.get("SPR_BENCH", {})

models = ["Baseline", "SymToken", "NoSymToken"]
colors = {"Baseline": "tab:blue", "SymToken": "tab:orange", "NoSymToken": "tab:green"}


# helper to pull series
def series(exp_dict, key):
    return [d[key] for d in exp_dict] if exp_dict else []


# ---------------------------------------------------------------- loss curves
try:
    plt.figure()
    for m in models:
        exp = bench.get(m, {})
        tloss = series(exp.get("losses", {}).get("train", []), "loss")
        vloss = series(exp.get("losses", {}).get("val", []), "loss")
        epochs = range(1, len(tloss) + 1)
        if tloss:
            plt.plot(epochs, tloss, "--", color=colors[m], label=f"{m} train")
        if vloss:
            plt.plot(epochs, vloss, "-", color=colors[m], label=f"{m} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------------------------------------------------------------- macro-F1 curves
try:
    plt.figure()
    for m in models:
        exp = bench.get(m, {})
        tf1 = series(exp.get("metrics", {}).get("train", []), "macro_f1")
        vf1 = series(exp.get("metrics", {}).get("val", []), "macro_f1")
        epochs = range(1, len(tf1) + 1)
        if tf1:
            plt.plot(epochs, tf1, "--", color=colors[m], label=f"{m} train")
        if vf1:
            plt.plot(epochs, vf1, "-", color=colors[m], label=f"{m} val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 Curves")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# ---------------------------------------------------------------- final RGA bar chart
try:
    plt.figure()
    fin_accs = []
    for m in models:
        vals = bench.get(m, {}).get("metrics", {}).get("val", [])
        fin_accs.append(vals[-1]["RGA"] if vals else 0.0)
    plt.bar(models, fin_accs, color=[colors[m] for m in models])
    plt.ylabel("Validation RGA")
    plt.ylim(0, 1)
    plt.title("SPR_BENCH Final Validation RGA")
    fname = os.path.join(working_dir, "SPR_BENCH_final_RGA.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating RGA bar chart: {e}")
    plt.close()

# ---------------------------------------------------------------- print summary metrics
for m, acc in zip(models, fin_accs):
    vf1 = bench.get(m, {}).get("metrics", {}).get("val", [])
    print(
        f'{m}: final val_F1={vf1[-1]["macro_f1"] if vf1 else None:.3f}, '
        f"val_RGA={acc:.3f}"
    )
