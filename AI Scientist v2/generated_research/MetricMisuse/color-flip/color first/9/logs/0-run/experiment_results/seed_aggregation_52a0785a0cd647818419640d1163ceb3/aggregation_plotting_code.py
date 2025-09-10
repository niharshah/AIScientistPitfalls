import matplotlib.pyplot as plt
import numpy as np
import os, math, itertools

# ---- basic IO set-up ---------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# list of paths supplied in the task description
experiment_data_path_list = [
    "experiments/2025-08-31_14-12-07_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_94f5f65338764b659a3277ccd95e1ed1_proc_1733531/experiment_data.npy",
    "None/experiment_data.npy",
    "experiments/2025-08-31_14-12-07_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_04f1d8dd3f5e4d16be4ce8eb0c9c01b6_proc_1733532/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    # skip obviously invalid entries
    if p is None or "None" in p:
        continue
    try:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        exp_data = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(exp_data)
    except Exception as e:
        print(f"Error loading experiment data {p}: {e}")


# ---- aggregation helpers -----------------------------------------------------
def append_nested(d, *keys, value):
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    lst = cur.setdefault(keys[-1], [])
    lst.append(value)


aggregated = {}  # dataset -> stuff

for exp in all_experiment_data:
    for dname, dct in exp.items():
        # losses
        for split in ("train", "val"):
            for lr, ep, val in dct.get("losses", {}).get(split, []):
                append_nested(aggregated, dname, "losses", split, lr, ep, value=val)
        # metrics (only val in most logs, test separately)
        for lr, ep, cwa, swa, hwa, *rest in dct.get("metrics", {}).get("val", []):
            append_nested(aggregated, dname, "metrics", "val", lr, ep, "cwa", value=cwa)
            append_nested(aggregated, dname, "metrics", "val", lr, ep, "swa", value=swa)
            append_nested(aggregated, dname, "metrics", "val", lr, ep, "hwa", value=hwa)
        # optional test metrics (single record per run)
        test_rec = dct.get("metrics", {}).get("test", None)
        if test_rec:
            lr, cwa, swa, hwa, *rest = test_rec
            append_nested(aggregated, dname, "metrics", "test", lr, "cwa", value=cwa)
            append_nested(aggregated, dname, "metrics", "test", lr, "swa", value=swa)
            append_nested(aggregated, dname, "metrics", "test", lr, "hwa", value=hwa)


# ---- plotting ---------------------------------------------------------------
def mean_sem(arr):
    arr = np.asarray(arr, dtype=float)
    m = np.nanmean(arr)
    if arr.size > 1:
        sem = np.nanstd(arr, ddof=1) / math.sqrt(arr.size)
    else:
        sem = 0.0
    return m, sem


for dname, dct in aggregated.items():
    # ---------------- loss curves --------------------------------------------
    try:
        plt.figure()
        max_ep = 1
        for lr, ep_dict in dct["losses"]["train"].items():
            max_ep = max(max_ep, max(ep_dict))
        stride = max(1, int(math.ceil(max_ep / 5)))
        for lr in sorted(dct["losses"]["train"]):
            # train
            ep_list = sorted(dct["losses"]["train"][lr])
            sel = ep_list[::stride] + (
                [ep_list[-1]] if ep_list[-1] not in ep_list[::stride] else []
            )
            y_train = []
            yerr_train = []
            for ep in sel:
                m, s = mean_sem(dct["losses"]["train"][lr][ep])
                y_train.append(m)
                yerr_train.append(s)
            plt.errorbar(
                sel, y_train, yerr=yerr_train, marker="o", label=f"train lr={lr:.0e}"
            )
            # val, if exists
            if lr in dct["losses"].get("val", {}):
                y_val = []
                yerr_val = []
                for ep in sel:
                    m, s = mean_sem(dct["losses"]["val"][lr].get(ep, [np.nan]))
                    y_val.append(m)
                    yerr_val.append(s)
                plt.errorbar(
                    sel,
                    y_val,
                    yerr=yerr_val,
                    marker="x",
                    linestyle="--",
                    label=f"val lr={lr:.0e}",
                )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname}: Mean ± SEM Training (solid) vs Validation (dashed) Loss")
        plt.legend()
        out = os.path.join(working_dir, f"{dname}_agg_loss_curves.png")
        plt.savefig(out)
        plt.close()
    except Exception as e:
        print(f"{dname}: aggregated loss plot error {e}")
        plt.close()

    # ---------------- validation HWA -----------------------------------------
    try:
        plt.figure()
        max_ep = 1
        for lr in dct["metrics"]["val"]:
            max_ep = max(max_ep, max(dct["metrics"]["val"][lr]))
        stride = max(1, int(math.ceil(max_ep / 5)))
        for lr in sorted(dct["metrics"]["val"]):
            ep_list = sorted(dct["metrics"]["val"][lr])
            sel = ep_list[::stride] + (
                [ep_list[-1]] if ep_list[-1] not in ep_list[::stride] else []
            )
            y = []
            yerr = []
            for ep in sel:
                m, s = mean_sem(dct["metrics"]["val"][lr][ep]["hwa"])
                y.append(m)
                yerr.append(s)
            plt.errorbar(sel, y, yerr=yerr, marker="o", label=f"lr={lr:.0e}")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title(f"{dname}: Mean ± SEM Validation Harmonic Weighted Accuracy")
        plt.legend()
        out = os.path.join(working_dir, f"{dname}_agg_val_hwa.png")
        plt.savefig(out)
        plt.close()
    except Exception as e:
        print(f"{dname}: aggregated HWA plot error {e}")
        plt.close()

    # ---------------- CWA vs SWA scatter (final epoch) ------------------------
    try:
        plt.figure()
        for lr in dct["metrics"]["val"]:
            last_ep = max(dct["metrics"]["val"][lr])
            cwa_m, cwa_s = mean_sem(dct["metrics"]["val"][lr][last_ep]["cwa"])
            swa_m, swa_s = mean_sem(dct["metrics"]["val"][lr][last_ep]["swa"])
            plt.errorbar(
                cwa_m,
                swa_m,
                xerr=cwa_s,
                yerr=swa_s,
                fmt="o",
                capsize=3,
                label=f"lr={lr:.0e}",
            )
        plt.xlabel("CWA")
        plt.ylabel("SWA")
        plt.title(f"{dname}: Final-Epoch Mean CWA vs SWA ± SEM")
        plt.legend()
        out = os.path.join(working_dir, f"{dname}_agg_cwa_swa_scatter.png")
        plt.savefig(out)
        plt.close()
    except Exception as e:
        print(f"{dname}: aggregated scatter plot error {e}")
        plt.close()

    # ---------------- test HWA bar -------------------------------------------
    try:
        plt.figure()
        lrs = []
        hwa_means = []
        hwa_sems = []
        source = dct["metrics"].get("test", None)
        if source:  # real test split
            for lr in source:
                m, s = mean_sem(source[lr]["hwa"])
                lrs.append(lr)
                hwa_means.append(m)
                hwa_sems.append(s)
        else:  # fall back to last val epoch
            for lr in dct["metrics"]["val"]:
                last_ep = max(dct["metrics"]["val"][lr])
                m, s = mean_sem(dct["metrics"]["val"][lr][last_ep]["hwa"])
                lrs.append(lr)
                hwa_means.append(m)
                hwa_sems.append(s)
        x = np.arange(len(lrs))
        plt.bar(x, hwa_means, yerr=hwa_sems, capsize=5)
        plt.xticks(x, [f"{lr:.0e}" for lr in lrs])
        plt.ylabel("HWA")
        plt.title(f"{dname}: Mean ± SEM Test (or Final-Val) HWA per LR")
        out = os.path.join(working_dir, f"{dname}_agg_test_hwa_bar.png")
        plt.savefig(out)
        plt.close()
    except Exception as e:
        print(f"{dname}: aggregated bar plot error {e}")
        plt.close()

    # ---------------- print summary ------------------------------------------
    try:
        print(f"\n--- {dname} aggregated results ---")
        for lr in sorted(dct["metrics"]["val"]):
            last_ep = max(dct["metrics"]["val"][lr])
            cwa_m, cwa_s = mean_sem(dct["metrics"]["val"][lr][last_ep]["cwa"])
            swa_m, swa_s = mean_sem(dct["metrics"]["val"][lr][last_ep]["swa"])
            hwa_m, hwa_s = mean_sem(dct["metrics"]["val"][lr][last_ep]["hwa"])
            print(
                f"lr={lr:.0e} | CWA: {cwa_m:.3f}±{cwa_s:.3f} "
                f"SWA: {swa_m:.3f}±{swa_s:.3f} HWA: {hwa_m:.3f}±{hwa_s:.3f}"
            )
    except Exception as e:
        print(f"{dname}: printing summary error {e}")
