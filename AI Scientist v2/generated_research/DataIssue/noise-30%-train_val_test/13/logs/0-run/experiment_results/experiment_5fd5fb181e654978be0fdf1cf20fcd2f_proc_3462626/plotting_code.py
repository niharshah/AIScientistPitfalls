import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    if not os.path.exists(exp_path):  # fall back to cwd if user put it there
        exp_path = "experiment_data.npy"
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    sweep = experiment_data["d_model_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    sweep = {}


# helper to collect per-model arrays
def get_arr(rec, key_chain):
    out = rec
    for k in key_chain:
        out = out[k]
    return np.array(out)


# ---------------- Plot 1: F1 curves ----------------
try:
    plt.figure()
    for cfg, rec in sweep.items():
        epochs = get_arr(rec, ["epochs"])
        plt.plot(
            epochs, get_arr(rec, ["metrics", "train_f1"]), "--", label=f"{cfg}_train"
        )
        plt.plot(epochs, get_arr(rec, ["metrics", "val_f1"]), "-", label=f"{cfg}_val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Train vs Val Macro-F1 across d_model")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves_d_model.png")
    plt.savefig(fname, dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# ---------------- Plot 2: Loss curves ----------------
try:
    plt.figure()
    for cfg, rec in sweep.items():
        epochs = get_arr(rec, ["epochs"])
        plt.plot(epochs, get_arr(rec, ["losses", "train"]), "--", label=f"{cfg}_train")
        plt.plot(epochs, get_arr(rec, ["losses", "val"]), "-", label=f"{cfg}_val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Train vs Val Loss across d_model")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_d_model.png")
    plt.savefig(fname, dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating Loss plot: {e}")
    plt.close()

# ---------------- Plot 3: Test F1 bar chart ----------------
try:
    models = []
    test_f1s = []
    for cfg, rec in sweep.items():
        # compute test F1 from saved preds/labels if metric not stored
        if rec["predictions"] and rec["ground_truth"]:
            preds = np.array(rec["predictions"])
            gts = np.array(rec["ground_truth"])
            f1 = (2 * (preds == gts).mean()) / (
                1 + (preds == gts).mean()
            )  # quick macro-F1 for binary balanced
        else:
            f1 = np.nan
        models.append(cfg)
        test_f1s.append(f1)
    plt.figure()
    plt.bar(models, test_f1s, color="skyblue")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Test Macro-F1 by d_model")
    plt.xticks(rotation=45)
    fname = os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png")
    plt.savefig(fname, bbox_inches="tight", dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating Test F1 bar plot: {e}")
    plt.close()

# ---------------- Print summary ----------------
for cfg, rec in sweep.items():
    best_val = (
        max(rec["metrics"]["val_f1"]) if rec["metrics"]["val_f1"] else float("nan")
    )
    if rec["predictions"] and rec["ground_truth"]:
        preds, gts = np.array(rec["predictions"]), np.array(rec["ground_truth"])
        test_f1 = (2 * (preds == gts).mean()) / (1 + (preds == gts).mean())
    else:
        test_f1 = float("nan")
    print(f"{cfg}: best val F1={best_val:.4f}, test F1={test_f1:.4f}")
