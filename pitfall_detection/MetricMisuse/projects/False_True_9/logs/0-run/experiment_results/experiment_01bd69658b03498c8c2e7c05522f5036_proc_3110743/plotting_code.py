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
    experiment_data = {}


# ---------- helper to unpack ----------
def unpack_losses(metrics_dict, kind="losses"):
    xs, ys_tr, ys_val = {}, {}, {}
    for hs, data in metrics_dict.items():
        rec = data["SPR_BENCH"][kind]
        if kind == "losses":
            tr = [t for t, _ in rec["train"]]
            ys_tr[hs] = [v for _, v in rec["train"]]
            ys_val[hs] = [v for _, v in rec["val"]]
            xs[hs] = tr
        else:  # metrics
            xs[hs] = [t for t, *_ in rec["val"]]
            swa = [s for _, s, _, _ in rec["val"]]
            cwa = [c for _, _, c, _ in rec["val"]]
            hwa = [h for _, _, _, h in rec["val"]]
            ys_tr[hs] = (swa, cwa, hwa)  # reuse container
    return xs, ys_tr, ys_val


# ---------- plotting ----------
for model_type in ["bidirectional", "unidirectional"]:
    configs = experiment_data.get(model_type, {})
    if not configs:
        continue

    # 1) Loss curves
    try:
        plt.figure()
        xs, ys_tr, ys_val = unpack_losses(configs, "losses")
        for hs in sorted(xs):
            plt.plot(xs[hs], ys_tr[hs], label=f"hs{hs}-train")
            plt.plot(xs[hs], ys_val[hs], "--", label=f"hs{hs}-val")
        plt.title(f"SPR_BENCH {model_type} LSTM - Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"SPR_BENCH_{model_type}_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating {model_type} loss plot: {e}")
        plt.close()

    # 2) HWA curves
    try:
        plt.figure()
        xs, ys_tr, _ = unpack_losses(configs, "metrics")
        for hs in sorted(xs):
            hwa = ys_tr[hs][2]  # index 2 is HWA
            plt.plot(xs[hs], hwa, label=f"hs{hs}")
        plt.title(f"SPR_BENCH {model_type} LSTM - Harmonic Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend()
        fname = f"SPR_BENCH_{model_type}_hwa_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating {model_type} HWA plot: {e}")
        plt.close()

# ---------- print best final metrics ----------
best_cfg = None
best_hwa = -1
for model_type, cfgs in experiment_data.items():
    for hs, data in cfgs.items():
        hwa_final = data["SPR_BENCH"]["metrics"]["val"][-1][-1]
        if hwa_final > best_hwa:
            best_hwa = hwa_final
            swa_final = data["SPR_BENCH"]["metrics"]["val"][-1][1]
            cwa_final = data["SPR_BENCH"]["metrics"]["val"][-1][2]
            best_cfg = (model_type, hs, swa_final, cwa_final, hwa_final)

if best_cfg:
    mtype, hs, swa_f, cwa_f, hwa_f = best_cfg
    print(
        f"Best final config: {mtype}, hidden={hs} "
        f"SWA={swa_f:.4f} CWA={cwa_f:.4f} HWA={hwa_f:.4f}"
    )
else:
    print("No data available to compute best metrics.")
