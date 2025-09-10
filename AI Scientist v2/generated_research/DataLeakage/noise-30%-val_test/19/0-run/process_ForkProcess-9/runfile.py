import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    wd_dict = experiment_data["weight_decay"]["SPR_BENCH"]
    wds = sorted(wd_dict.keys(), key=lambda x: float(x))
    # ------------------------------------------------
    # 1) train-F1 curves
    try:
        plt.figure(figsize=(6, 4))
        for wd in wds:
            rec = wd_dict[wd]
            plt.plot(rec["epochs"], rec["metrics"]["train_macro_f1"], label=f"wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("Training Macro-F1 vs Epoch (SPR_BENCH)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_train_macro_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating train-F1 plot: {e}")
        plt.close()
    # ------------------------------------------------
    # 2) val-F1 curves
    try:
        plt.figure(figsize=(6, 4))
        for wd in wds:
            rec = wd_dict[wd]
            plt.plot(rec["epochs"], rec["metrics"]["val_macro_f1"], label=f"wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("Validation Macro-F1 vs Epoch (SPR_BENCH)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_val_macro_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating val-F1 plot: {e}")
        plt.close()
    # ------------------------------------------------
    # 3) final test-F1 bar chart
    try:
        test_f1s = [
            np.mean(wd_dict[wd]["metrics"]["val_macro_f1"][-1:]) * 0 for wd in wds
        ]  # dummy init
        # recover stored test scores (they are not in metrics, so rebuild)
        test_f1s = []
        for wd in wds:
            preds = np.array(wd_dict[wd]["predictions"])
            gt = np.array(wd_dict[wd]["ground_truth"])
            # recompute macro-F1 just for plotting (SAFETY)
            from sklearn.metrics import f1_score

            test_f1 = f1_score(gt, preds, average="macro") if preds.size else 0.0
            test_f1s.append(test_f1)
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(wds)), test_f1s, tick_label=wds)
        plt.ylabel("Macro-F1")
        plt.xlabel("Weight Decay")
        plt.title("Final Test Macro-F1 vs Weight Decay (SPR_BENCH)")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_macro_f1_bars.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test-F1 bar chart: {e}")
        plt.close()
    # ------------------------------------------------
    # 4) loss curve for best run
    try:
        best_wd = max(wds, key=lambda wd: wd_dict[wd]["metrics"]["val_macro_f1"][-1])
        rec = wd_dict[best_wd]
        plt.figure(figsize=(6, 4))
        plt.plot(rec["epochs"], rec["losses"]["train"], label="train_loss")
        plt.plot(rec["epochs"], rec["losses"]["val"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves (SPR_BENCH, best wd={best_wd})")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(
            working_dir, f"SPR_BENCH_loss_curves_best_wd_{best_wd}.png"
        )
        plt.savefig(fname)
        plt.close()
        # print best score
        from sklearn.metrics import f1_score

        preds = np.array(rec["predictions"])
        gt = np.array(rec["ground_truth"])
        best_test_f1 = f1_score(gt, preds, average="macro") if preds.size else 0.0
        print(f"Best weight_decay={best_wd} | Test Macro-F1={best_test_f1:.4f}")
    except Exception as e:
        print(f"Error creating best-run loss curve: {e}")
        plt.close()
