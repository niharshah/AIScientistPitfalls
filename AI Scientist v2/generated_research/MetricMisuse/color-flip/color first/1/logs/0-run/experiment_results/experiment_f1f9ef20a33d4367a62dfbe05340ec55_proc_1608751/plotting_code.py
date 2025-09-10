import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
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

tags = list(experiment_data.keys())
if not tags:
    print("No experiment data found, nothing to plot.")
else:
    # pre-extract useful arrays
    epochs, tr_loss, va_loss, va_hmwa, tst_hmwa, tst_cwa, tst_swa = (
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )
    for tag in tags:
        ed = experiment_data[tag]["SPR_BENCH"]
        tr_loss[tag] = ed["losses"]["train"]
        va_loss[tag] = ed["losses"]["val"]
        va_hmwa[tag] = [m["hmwa"] for m in ed["metrics"]["val"]]
        tst = ed["metrics"]["test"]
        tst_hmwa[tag], tst_cwa[tag], tst_swa[tag] = tst["hmwa"], tst["cwa"], tst["swa"]
        epochs[tag] = list(range(1, len(tr_loss[tag]) + 1))

    # ---------- plot 1 : loss curves ----------
    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for tag in tags:
            ax[0].plot(epochs[tag], tr_loss[tag], label=tag)
            ax[1].plot(epochs[tag], va_loss[tag], label=tag)
        ax[0].set_title("Train Loss")
        ax[1].set_title("Validation Loss")
        for a in ax:
            a.set_xlabel("Epoch")
            a.set_ylabel("CE Loss")
            a.legend()
        fig.suptitle("SPR_BENCH Loss Curves (Left: Train, Right: Validation)")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plots: {e}")
        plt.close()

    # ---------- plot 2 : validation HMWA ----------
    try:
        plt.figure(figsize=(6, 4))
        for tag in tags:
            plt.plot(epochs[tag], va_hmwa[tag], label=tag)
        plt.title("SPR_BENCH Validation HMWA over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("HMWA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_HMWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating HMWA plot: {e}")
        plt.close()

    # ---------- plot 3 : test HMWA bar ----------
    try:
        plt.figure(figsize=(6, 4))
        names, scores = zip(*tst_hmwa.items())
        plt.bar(names, scores, color="skyblue")
        plt.title("SPR_BENCH Test HMWA by Run")
        plt.ylabel("HMWA")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_HMWA_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating HMWA bar: {e}")
        plt.close()

    # ---------- plot 4 : CWA vs SWA scatter ----------
    try:
        plt.figure(figsize=(5, 5))
        for tag in tags:
            plt.scatter(tst_cwa[tag], tst_swa[tag], label=tag)
            plt.text(tst_cwa[tag], tst_swa[tag], tag)
        plt.title("SPR_BENCH Test CWA vs SWA")
        plt.xlabel("CWA")
        plt.ylabel("SWA")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_CWA_vs_SWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CWA-SWA scatter: {e}")
        plt.close()

    # ---------- plot 5 : confusion matrix (first tag) ----------
    first_tag = tags[0]
    try:
        preds = experiment_data[first_tag]["SPR_BENCH"].get("predictions", [])
        golds = experiment_data[first_tag]["SPR_BENCH"].get("ground_truth", [])
        if preds and golds:
            num_cls = len(set(golds))
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for g, p in zip(golds, preds):
                cm[g, p] += 1
            plt.figure(figsize=(5, 4))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{first_tag} Confusion Matrix (Test)")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{first_tag}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------- print summary ----------
    print("\nFinal Test Metrics:")
    for tag in tags:
        print(
            f"{tag}: CWA={tst_cwa[tag]:.4f} | SWA={tst_swa[tag]:.4f} | HMWA={tst_hmwa[tag]:.4f}"
        )
