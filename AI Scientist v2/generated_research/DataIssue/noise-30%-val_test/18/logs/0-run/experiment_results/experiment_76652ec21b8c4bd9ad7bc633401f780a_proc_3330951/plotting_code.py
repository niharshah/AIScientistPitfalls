import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    embed_dict = experiment_data.get("embed_dim", {})
    dims = sorted(embed_dict.keys())
    # --------- collect per-dim arrays ----------
    f1_curves, loss_curves = {}, {}
    best_f1s, best_dim = [], None
    overall_best = -1
    for d in dims:
        ed = embed_dict[d]
        f1_curves[d] = ed["metrics"]["val"]
        loss_curves[d] = ed["losses"]
        final_f1 = ed["metrics"]["val"][-1] if ed["metrics"]["val"] else np.nan
        best_f1s.append(final_f1)
        if final_f1 > overall_best:
            overall_best = final_f1
            best_dim = d

    # ---------- 1: Macro-F1 curves ----------
    try:
        plt.figure()
        for d in dims:
            plt.plot(embed_dict[d]["epochs"], f1_curves[d], label=f"dim={d}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Validation Macro-F1 vs Epoch (per embedding dim)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve plot: {e}")
        plt.close()

    # ---------- 2: Loss curves ----------
    try:
        plt.figure()
        for d in dims:
            plt.plot(
                embed_dict[d]["epochs"],
                loss_curves[d]["train"],
                "--",
                label=f"train dim={d}",
            )
            plt.plot(
                embed_dict[d]["epochs"], loss_curves[d]["val"], label=f"val dim={d}"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train/Val Loss vs Epoch (all embedding dims)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- 3: Best F1 per dim ----------
    try:
        plt.figure()
        plt.bar([str(d) for d in dims], best_f1s, color="skyblue")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Best Validation Macro-F1")
        plt.title("SPR_BENCH: Best Dev Macro-F1 per Embedding Dim")
        plt.savefig(os.path.join(working_dir, "spr_bench_best_f1_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating bar plot: {e}")
        plt.close()

    # ---------- 4: Confusion Matrix for best dim ----------
    try:
        from sklearn.metrics import confusion_matrix

        preds = embed_dict[best_dim]["predictions"]
        gts = embed_dict[best_dim]["ground_truth"]
        cm = confusion_matrix(gts, preds, labels=[0, 1])
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"SPR_BENCH: Confusion Matrix (Best dim={best_dim})")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        plt.savefig(os.path.join(working_dir, f"spr_bench_confmat_dim{best_dim}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()
