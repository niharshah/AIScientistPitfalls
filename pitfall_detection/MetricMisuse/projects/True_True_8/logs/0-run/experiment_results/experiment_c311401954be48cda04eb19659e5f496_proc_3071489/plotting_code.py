import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    losses_tr = experiment_data["batch_size"]["SPR_BENCH"]["losses"]["train"]
    losses_va = experiment_data["batch_size"]["SPR_BENCH"]["losses"]["val"]
    acs_va = experiment_data["batch_size"]["SPR_BENCH"]["metrics"]["val"]
    preds_all = experiment_data["batch_size"]["SPR_BENCH"]["predictions"]
    gts_all = experiment_data["batch_size"]["SPR_BENCH"]["ground_truth"]

    # collect unique batch sizes
    bs_list = sorted({bs for bs, _, _ in losses_tr})

    # -------- plot 1 : loss curves --------
    try:
        plt.figure(figsize=(6, 4))
        for bs in bs_list:
            x_tr = [ep for b, ep, _ in losses_tr if b == bs]
            y_tr = [vl for b, _, vl in losses_tr if b == bs]
            x_va = [ep for b, ep, _ in losses_va if b == bs]
            y_va = [vl for b, _, vl in losses_va if b == bs]
            plt.plot(x_tr, y_tr, label=f"train bs={bs}")
            plt.plot(x_va, y_va, "--", label=f"val bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------- plot 2 : ACS curves --------
    try:
        plt.figure(figsize=(6, 4))
        for bs in bs_list:
            x_acs = [ep for b, ep, _ in acs_va if b == bs]
            y_acs = [vl for b, _, vl in acs_va if b == bs]
            plt.plot(x_acs, y_acs, marker="o", label=f"bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("ACS")
        plt.title("SPR_BENCH: Validation ACS Across Epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_ACS_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating ACS plot: {e}")
        plt.close()

    # -------- plot 3 : accuracy per batch size --------
    try:
        # compute accuracy
        acc_dict = {bs: [0, 0] for bs in bs_list}  # correct, total
        for (bs, pred), gt in zip(preds_all, gts_all):
            acc_dict[bs][1] += 1
            if pred == gt:
                acc_dict[bs][0] += 1
        acc_pct = {bs: (c / t if t else 0.0) for bs, (c, t) in acc_dict.items()}

        plt.figure(figsize=(6, 4))
        plt.bar(list(acc_pct.keys()), list(acc_pct.values()), color="skyblue")
        plt.ylim(0, 1)
        plt.xlabel("Batch Size")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Final Validation Accuracy by Batch Size")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_val_accuracy_by_batch_size.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ---------- print evaluation metric ----------
    print(
        "Validation accuracy per batch size:",
        {bs: round(acc_pct[bs], 4) for bs in bs_list},
    )
