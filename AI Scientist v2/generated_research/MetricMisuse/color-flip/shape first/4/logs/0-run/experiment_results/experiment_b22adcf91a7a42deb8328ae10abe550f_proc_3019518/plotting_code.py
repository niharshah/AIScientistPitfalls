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
    ed = experiment_data["weight_decay"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed:
    # rebuild per-weight-decay series
    epochs_per_wd = 5
    wds = sorted(set(ed["hyperparams"]))
    losses_train, losses_val, hwas = {}, {}, {}
    for i, wd in enumerate(ed["hyperparams"]):
        idx = wds.index(wd)
        step = i % epochs_per_wd
        losses_train.setdefault(wd, []).append(ed["losses"]["train"][i])
        losses_val.setdefault(wd, []).append(ed["losses"]["val"][i])
        hwas.setdefault(wd, []).append(ed["metrics"]["val"][i])

    # 1) Loss curves
    try:
        plt.figure(figsize=(8, 4))
        for wd in wds:
            plt.plot(losses_train[wd], "--", label=f"train wd={wd}")
            plt.plot(losses_val[wd], "-", label=f"val wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            "SPR_BENCH Training and Validation Loss vs Epoch\n(weight_decay sweep)"
        )
        plt.legend(fontsize=8)
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # 2) HWA curves
    try:
        plt.figure(figsize=(6, 4))
        for wd in wds:
            plt.plot(hwas[wd], label=f"wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy")
        plt.title("SPR_BENCH HWA vs Epoch (weight_decay sweep)")
        plt.legend(fontsize=8)
        fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curves: {e}")
        plt.close()

    # 3) Final HWA by WD
    try:
        final_hwa = [hwas[wd][-1] for wd in wds]
        plt.figure(figsize=(5, 4))
        plt.plot(wds, final_hwa, "o-")
        plt.xscale("log")
        plt.xlabel("Weight Decay")
        plt.ylabel("Final Epoch HWA")
        plt.title("SPR_BENCH Final HWA vs Weight Decay")
        fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_vs_wd.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating final HWA plot: {e}")
        plt.close()

    # Print summary table
    print("Final HWA per weight_decay:")
    for wd, score in zip(wds, final_hwa):
        print(f"  wd={wd}: HWA={score:.3f}")
