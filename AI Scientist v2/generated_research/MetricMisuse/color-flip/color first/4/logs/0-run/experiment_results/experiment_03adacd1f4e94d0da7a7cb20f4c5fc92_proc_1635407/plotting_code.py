import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    bs_tags = sorted(
        experiment_data["batch_size"].keys(), key=lambda t: int(t.split("_")[1])
    )  # ['bs_32', ...]
    epochs = range(
        1,
        1
        + len(
            experiment_data["batch_size"][bs_tags[0]]["SPR_BENCH"]["losses"]["train"]
        ),
    )

    # -------------------- 1) Loss curves -------------------- #
    try:
        plt.figure(figsize=(6, 4))
        for tag in bs_tags:
            tr = experiment_data["batch_size"][tag]["SPR_BENCH"]["losses"]["train"]
            vl = experiment_data["batch_size"][tag]["SPR_BENCH"]["losses"]["val"]
            plt.plot(epochs, tr, label=f"{tag} train", linestyle="-")
            plt.plot(epochs, vl, label=f"{tag} val", linestyle="--")
        plt.title("SPR_BENCH: Train vs Val Loss (all batch sizes)")
        plt.xlabel("Epoch")
        plt.ylabel("CE Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # -------------------- 2) Validation ACC -------------------- #
    try:
        plt.figure(figsize=(6, 4))
        for tag in bs_tags:
            acc = [
                m["acc"]
                for m in experiment_data["batch_size"][tag]["SPR_BENCH"]["metrics"][
                    "val"
                ]
            ]
            plt.plot(epochs, acc, marker="o", label=tag)
        plt.title("SPR_BENCH: Validation Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_val_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating val accuracy plot: {e}")
        plt.close()

    # -------------------- 3) Validation PCWA -------------------- #
    try:
        plt.figure(figsize=(6, 4))
        for tag in bs_tags:
            pcwa = [
                m["pcwa"]
                for m in experiment_data["batch_size"][tag]["SPR_BENCH"]["metrics"][
                    "val"
                ]
            ]
            plt.plot(epochs, pcwa, marker="s", label=tag)
        plt.title("SPR_BENCH: Validation PC-Weighted Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("PCWA")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_val_pcwa.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating pcwa plot: {e}")
        plt.close()

    # -------------------- 4) Final ACC bar -------------------- #
    try:
        final_accs = [
            experiment_data["batch_size"][tag]["SPR_BENCH"]["metrics"]["val"][-1]["acc"]
            for tag in bs_tags
        ]
        plt.figure(figsize=(6, 4))
        plt.bar(bs_tags, final_accs, color="skyblue")
        plt.title("SPR_BENCH: Final Validation Accuracy by Batch Size")
        plt.ylabel("Accuracy")
        fname = os.path.join(working_dir, "spr_bench_final_acc_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final acc bar: {e}")
        plt.close()

    # -------------------- 5) Confusion Matrix -------------------- #
    try:
        # choose batch size with best final acc
        best_idx = int(np.argmax(final_accs))
        best_tag = bs_tags[best_idx]
        preds = experiment_data["batch_size"][best_tag]["SPR_BENCH"]["predictions"]
        gts = experiment_data["batch_size"][best_tag]["SPR_BENCH"]["ground_truth"]
        num_cls = max(max(preds), max(gts)) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for g, p in zip(gts, preds):
            cm[g, p] += 1
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.title(f"SPR_BENCH: Confusion Matrix (best={best_tag})")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.xticks(range(num_cls))
        plt.yticks(range(num_cls))
        fname = os.path.join(working_dir, f"spr_bench_confusion_{best_tag}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
