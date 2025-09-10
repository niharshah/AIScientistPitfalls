import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper ----------
def save_plot(fig_name):
    fname = os.path.join(working_dir, fig_name)
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()


for dname, run in experiment_data.items():
    # ---- 1. pre-training loss ----
    try:
        losses = run.get("losses", {}).get("pretrain", [])
        if losses:
            plt.figure()
            plt.plot(range(1, len(losses) + 1), losses, label="Pre-train")
            plt.title(f"{dname}: Pre-training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            save_plot(f"{dname}_pretrain_loss.png")
    except Exception as e:
        print(f"Error plotting pretrain loss for {dname}: {e}")
        plt.close()

    # ---- 2. fine-tuning loss ----
    try:
        tr = run.get("losses", {}).get("train", [])
        vl = run.get("losses", {}).get("val", [])
        if tr or vl:
            plt.figure()
            if tr:
                plt.plot(range(1, len(tr) + 1), tr, label="Train")
            if vl:
                plt.plot(range(1, len(vl) + 1), vl, label="Val")
            plt.title(f"{dname}: Fine-tuning Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            save_plot(f"{dname}_finetune_loss.png")
    except Exception as e:
        print(f"Error plotting finetune loss for {dname}: {e}")
        plt.close()

    # ---- 3-5. metric curves (at most three) ----
    metric_list = ["SWA", "CWA", "CompWA"]
    metrics_per_epoch = run.get("metrics", {}).get("val", [])
    for met in metric_list:
        try:
            vals = [ep.get(met) for ep in metrics_per_epoch if ep.get(met) is not None]
            if vals:
                plt.figure()
                plt.plot(range(1, len(vals) + 1), vals, label=met)
                plt.title(f"{dname}: {met} (Validation)")
                plt.xlabel("Fine-tuning Epoch")
                plt.ylabel(met)
                plt.legend()
                save_plot(f"{dname}_{met}_curve.png")
        except Exception as e:
            print(f"Error plotting {met} for {dname}: {e}")
            plt.close()

    # ---- simple final accuracy printout ----
    try:
        preds = np.array(run.get("predictions", []))
        gtruth = np.array(run.get("ground_truth", []))
        if len(preds) == len(gtruth) and len(preds):
            acc = (preds == gtruth).mean()
            print(f"{dname}: final accuracy = {acc:.3f}")
    except Exception as e:
        print(f"Error computing accuracy for {dname}: {e}")
