import matplotlib.pyplot as plt
import numpy as np
import os, itertools, math

# ---------------- basic setup ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------- load all runs ---------------
experiment_data_path_list = [
    "experiments/2025-08-31_14-12-07_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_ed8257394b7347cdb6fa0059c8b6e570_proc_1748866/experiment_data.npy",
    "experiments/2025-08-31_14-12-07_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_95bebe314e85434aa10b728ea3f174a3_proc_1748869/experiment_data.npy",
    "experiments/2025-08-31_14-12-07_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_28d718de99a448ed93ab846384611a59_proc_1748868/experiment_data.npy",
]
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        all_experiment_data.append(np.load(full_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# ---------- helper functions ----------------
def reindex_loss(loss_list):
    d = {}
    for lr, ep, val in loss_list:
        d.setdefault(lr, {}).setdefault(ep, []).append(val)
    return d


def reindex_hwa(metric_list):
    d = {}
    for lr, ep, _cwa, _swa, hwa, *rest in metric_list:
        d.setdefault(lr, {}).setdefault(ep, []).append(hwa)
    return d


# ----------- aggregate across runs ----------
aggregate = {}  # {dname: {"train_loss":dict, "val_loss":dict, "val_hwa":dict}}
for run_data in all_experiment_data:
    for dname, dct in run_data.items():
        agg = aggregate.setdefault(
            dname, {"train_loss": {}, "val_loss": {}, "val_hwa": {}}
        )

        # collect losses
        for split, key in [("train_loss", "train"), ("val_loss", "val")]:
            for lr, ep_dict in reindex_loss(dct["losses"].get(key, [])).items():
                for ep, vals in ep_dict.items():
                    agg[split].setdefault(lr, {}).setdefault(ep, []).extend(vals)

        # collect validation HWA
        for lr, ep_dict in reindex_hwa(dct["metrics"].get("val", [])).items():
            for ep, vals in ep_dict.items():
                agg["val_hwa"].setdefault(lr, {}).setdefault(ep, []).extend(vals)

# ------------- plotting per dataset ----------
for dname, dct in aggregate.items():
    # determine stride so we never exceed 5 points
    all_epochs = set(
        itertools.chain.from_iterable(
            ep_dict.keys() for ep_dict in dct["train_loss"].values()
        )
    )
    max_ep = max(all_epochs) if all_epochs else 1
    stride = max(1, math.ceil(max_ep / 5))

    # ---------- Plot 1: aggregated loss curves ----------
    try:
        plt.figure()
        for lr, ep_dict in dct["train_loss"].items():
            eps = sorted(ep_dict.keys())
            sel = eps[::stride] + ([eps[-1]] if eps[-1] not in eps[::stride] else [])
            means = [np.mean(ep_dict[e]) for e in sel]
            stderrs = [
                np.std(ep_dict[e], ddof=1) / math.sqrt(len(ep_dict[e])) for e in sel
            ]
            plt.errorbar(
                sel, means, yerr=stderrs, fmt="-o", label=f"train lr={lr} (±SE)"
            )
            # validation
            v_ep_dict = dct["val_loss"].get(lr, {})
            if v_ep_dict:
                means_val = [np.mean(v_ep_dict.get(e, [np.nan])) for e in sel]
                stderrs_val = [
                    (
                        np.std(v_ep_dict.get(e, [np.nan]), ddof=1)
                        / math.sqrt(len(v_ep_dict.get(e, [np.nan])))
                        if not np.isnan(means_val[i])
                        else 0
                    )
                    for i, e in enumerate(sel)
                ]
                plt.errorbar(
                    sel,
                    means_val,
                    yerr=stderrs_val,
                    fmt="--x",
                    label=f"val lr={lr} (±SE)",
                )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname}: Training vs Validation Loss (mean ± SE)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_aggregated_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"{dname}: aggregated loss plot error {e}")
        plt.close()

    # ---------- Plot 2: aggregated HWA curves ----------
    try:
        plt.figure()
        for lr, ep_dict in dct["val_hwa"].items():
            eps = sorted(ep_dict.keys())
            sel = eps[::stride] + ([eps[-1]] if eps[-1] not in eps[::stride] else [])
            means = [np.mean(ep_dict[e]) for e in sel]
            stderrs = [
                np.std(ep_dict[e], ddof=1) / math.sqrt(len(ep_dict[e])) for e in sel
            ]
            plt.errorbar(sel, means, yerr=stderrs, fmt="-o", label=f"lr={lr} (±SE)")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title(f"{dname}: Validation Harmonic Weighted Accuracy (mean ± SE)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_aggregated_val_hwa.png"))
        plt.close()
    except Exception as e:
        print(f"{dname}: aggregated HWA plot error {e}")
        plt.close()
