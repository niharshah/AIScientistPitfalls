import matplotlib.pyplot as plt
import numpy as np
import os

# create / verify working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

bench_key = "SPR_BENCH"
root = experiment_data.get("NoPositionalEmbedding", {}).get(bench_key, {})

val_f1_compare = {}  # for final overlay plot

for model_name, logs in root.items():
    # --------- gather data ----------
    tr_loss = [(d["epoch"], d["loss"]) for d in logs["losses"]["train"]]
    val_loss = [(d["epoch"], d["loss"]) for d in logs["losses"]["val"]]
    tr_f1 = [(d["epoch"], d["macro_f1"]) for d in logs["metrics"]["train"]]
    val_f1 = [(d["epoch"], d["macro_f1"]) for d in logs["metrics"]["val"]]
    # save val f1 curve for comparison plot
    val_f1_compare[model_name] = val_f1

    # --------- loss curve ----------
    try:
        plt.figure()
        if tr_loss:
            ep, loss = zip(*tr_loss)
            plt.plot(ep, loss, label="Train")
        if val_loss:
            ep_v, loss_v = zip(*val_loss)
            plt.plot(ep_v, loss_v, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH {model_name} Loss Curve")
        plt.legend()
        fname = f"SPR_BENCH_{model_name}_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {model_name}: {e}")
        plt.close()

    # --------- F1 curve ----------
    try:
        plt.figure()
        if tr_f1:
            ep, f1 = zip(*tr_f1)
            plt.plot(ep, f1, label="Train")
        if val_f1:
            ep_v, f1_v = zip(*val_f1)
            plt.plot(ep_v, f1_v, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"SPR_BENCH {model_name} Macro-F1 Curve")
        plt.legend()
        fname = f"SPR_BENCH_{model_name}_f1_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot for {model_name}: {e}")
        plt.close()

    # --------- print final metrics ----------
    if logs["metrics"]["val"]:
        last = logs["metrics"]["val"][-1]
        print(
            f"{model_name} | Final Val Macro-F1: {last['macro_f1']:.3f} | RGA(Acc): {last['RGA']:.3f}"
        )

# --------- comparison plot ----------
try:
    if val_f1_compare:
        plt.figure()
        for model_name, curve in val_f1_compare.items():
            ep, f1 = zip(*curve)
            plt.plot(ep, f1, label=model_name)
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Validation Macro-F1 Comparison")
        plt.legend()
        fname = "SPR_BENCH_val_macro_f1_comparison.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating comparison plot: {e}")
    plt.close()
