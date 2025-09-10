import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

strategies = list(experiment_data.keys())
dataset = "SPR_BENCH"  # only dataset stored by the training script

# helper to pick color styles
colors = {"mean_pool": "tab:blue", "cls_token": "tab:orange"}

# ------------------------------------------------------------------
# 1) Pre-training loss curves (both strategies on same plot)
try:
    plt.figure()
    for strat in strategies:
        losses = experiment_data[strat][dataset]["losses"].get("pretrain", [])
        if losses:
            plt.plot(
                range(1, len(losses) + 1),
                losses,
                marker="o",
                label=strat,
                color=colors.get(strat, None),
            )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Pre-training Loss (mean_pool vs cls_token)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_pretrain_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating pretrain loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2-3) Fine-tuning train/val losses per strategy
for strat in strategies:
    try:
        plt.figure()
        losses = experiment_data[strat][dataset]["losses"]
        tr, val = losses.get("train", []), losses.get("val", [])
        epochs = range(1, max(len(tr), len(val)) + 1)
        if tr:
            plt.plot(epochs[: len(tr)], tr, marker="o", label="train", color="tab:blue")
        if val:
            plt.plot(
                epochs[: len(val)],
                val,
                marker="o",
                label="validation",
                color="tab:green",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"SPR_BENCH Fine-tuning Loss ({strat})")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_finetune_loss_{strat}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating fine-tune loss plot ({strat}): {e}")
        plt.close()

# ------------------------------------------------------------------
# 4) Validation SCWA curves (both strategies on same plot)
try:
    plt.figure()
    for strat in strategies:
        scwa = experiment_data[strat][dataset]["metrics"].get("val_SCWA", [])
        if scwa:
            plt.plot(
                range(1, len(scwa) + 1),
                scwa,
                marker="o",
                label=strat,
                color=colors.get(strat, None),
            )
    plt.xlabel("Epoch")
    plt.ylabel("SCWA")
    plt.title("SPR_BENCH: Validation SCWA (mean_pool vs cls_token)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_SCWA.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SCWA plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 5) Confusion matrix for best-performing strategy
# pick strategy with highest max SCWA
best_strat = None
best_score = -1
for strat in strategies:
    scwa_list = experiment_data[strat][dataset]["metrics"].get("val_SCWA", [])
    if scwa_list and max(scwa_list) > best_score:
        best_score = max(scwa_list)
        best_strat = strat

if best_strat:
    try:
        preds = np.array(experiment_data[best_strat][dataset]["predictions"])
        trues = np.array(experiment_data[best_strat][dataset]["ground_truth"])
        if preds.size and trues.size and preds.size == trues.size:
            num_labels = int(max(trues.max(), preds.max()) + 1)
            cm = np.zeros((num_labels, num_labels), dtype=int)
            for t, p in zip(trues, preds):
                cm[t, p] += 1

            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"SPR_BENCH Confusion Matrix (best: {best_strat})")
            # annotate cells (optional readability for small matrices)
            for i in range(num_labels):
                for j in range(num_labels):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
            fname = os.path.join(working_dir, f"SPR_BENCH_confusion_{best_strat}.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

# ------------------------------------------------------------------
# Print final evaluation metrics
for strat in strategies:
    mets = experiment_data[strat][dataset]["metrics"]
    swa, cwa, scwa = map(
        max,
        (mets.get("val_SWA", [0]), mets.get("val_CWA", [0]), mets.get("val_SCWA", [0])),
    )
    print(f"{strat}: best_SWA={swa:.4f}, best_CWA={cwa:.4f}, best_SCWA={scwa:.4f}")
