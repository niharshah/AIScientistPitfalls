import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data -------------------
try:
    ed = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = ed["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    # ------- reshape helpers -------
    tr_loss = {}
    val_loss = {}
    val_metrics = {}
    for lr, ep, loss in ed["losses"]["train"]:
        tr_loss.setdefault(lr, {})[ep] = loss
    for lr, ep, loss in ed["losses"]["val"]:
        val_loss.setdefault(lr, {})[ep] = loss
    for lr, ep, cwa, swa, hwa in ed["metrics"]["val"]:
        val_metrics.setdefault(lr, {})[ep] = (cwa, swa, hwa)

    test_res = {}
    if "test" in ed["metrics"]:
        lr, cwa, swa, hwa = ed["metrics"]["test"]
        test_res[lr] = (cwa, swa, hwa)

    max_ep = max(ep for lr in tr_loss for ep in tr_loss[lr])
    stride = max(1, int(np.ceil(max_ep / 5)))  # plot at most 5 points

    # ---------------- Plot 1: loss curves ----------------
    try:
        plt.figure()
        for lr in tr_loss:
            eps = sorted(tr_loss[lr])
            sel = eps[::stride] + ([eps[-1]] if eps[-1] not in eps[::stride] else [])
            plt.plot(sel, [tr_loss[lr][e] for e in sel], "-o", label=f"train lr={lr}")
            plt.plot(sel, [val_loss[lr][e] for e in sel], "--x", label=f"val lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------------- Plot 2: HWA curves -------------------
    try:
        plt.figure()
        for lr in val_metrics:
            eps = sorted(val_metrics[lr])
            sel = eps[::stride] + ([eps[-1]] if eps[-1] not in eps[::stride] else [])
            plt.plot(sel, [val_metrics[lr][e][2] for e in sel], "-o", label=f"lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("SPR_BENCH: Validation Harmonic Weighted Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_hwa.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # ------------- Plot 3: CWA vs SWA scatter ------------
    try:
        plt.figure()
        for lr in val_metrics:
            # use last epoch for each lr
            last_ep = max(val_metrics[lr])
            cwa, swa, _ = val_metrics[lr][last_ep]
            plt.scatter(cwa, swa, label=f"lr={lr}")
            plt.text(cwa, swa, f"{lr:.0e}")
        plt.xlabel("CWA")
        plt.ylabel("SWA")
        plt.title("SPR_BENCH: Final Epoch CWA vs SWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_cwa_swa_scatter.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        plt.close()

    # ------------- Plot 4: Test HWA bar ------------------
    try:
        plt.figure()
        if not test_res:  # if missing, synthesize from val last epoch
            for lr in val_metrics:
                last_ep = max(val_metrics[lr])
                test_res[lr] = val_metrics[lr][last_ep]
        lrs = list(test_res)
        hwas = [test_res[lr][2] for lr in lrs]
        plt.bar(range(len(lrs)), hwas, tick_label=[f"{lr:.0e}" for lr in lrs])
        plt.ylabel("HWA")
        plt.title("SPR_BENCH: Test Harmonic Weighted Accuracy by LR")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_hwa_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating bar plot: {e}")
        plt.close()
