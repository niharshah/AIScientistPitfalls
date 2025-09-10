import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ds_name = "SPR_BENCH"
    per_layer = experiment_data["num_layers"][ds_name]["per_layer"]
    colors = {nl: c for nl, c in zip(sorted(per_layer), ["r", "g", "b", "m", "c"])}

    # ------------------------------------------------- Plot 1: loss curves
    try:
        plt.figure()
        for nl, rec in per_layer.items():
            epochs = rec["epochs"]
            plt.plot(
                epochs,
                rec["losses"]["train"],
                linestyle="--",
                color=colors[nl],
                label=f"train L{nl}",
            )
            plt.plot(
                epochs,
                rec["losses"]["val"],
                linestyle="-",
                color=colors[nl],
                label=f"val L{nl}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title(f"{ds_name}: Training & Validation Loss vs Epoch")
        plt.legend()
        fn = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ------------------------------------------------- Plot 2: val SCWA curves
    try:
        plt.figure()
        for nl, rec in per_layer.items():
            epochs = rec["epochs"]
            plt.plot(epochs, rec["metrics"]["val"], color=colors[nl], label=f"L{nl}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation SCWA")
        plt.title(f"{ds_name}: Validation SCWA vs Epoch")
        plt.legend()
        fn = os.path.join(working_dir, f"{ds_name}_val_scwa_curves.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating SCWA curve plot: {e}")
        plt.close()

    # ------------------------------------------------- Plot 3: best SCWA summary
    try:
        best_val_scores = [rec["best_val_scwa"] for rec in per_layer.values()]
        layers = list(per_layer.keys())
        best_layer = experiment_data["num_layers"][ds_name]["best_layer"]
        test_scwa = experiment_data["num_layers"][ds_name]["test_scwa"]
        x = np.arange(len(layers))
        plt.figure()
        plt.bar(
            x,
            best_val_scores,
            color=[colors[l] for l in layers],
            alpha=0.7,
            label="Best Val SCWA",
        )
        # overlay test score of best layer
        plt.bar(
            layers.index(best_layer),
            test_scwa,
            color="k",
            alpha=0.4,
            label="Test SCWA (best layer)",
        )
        plt.xticks(x, [f"L{l}" for l in layers])
        plt.ylabel("SCWA")
        plt.title(f"{ds_name}: Best Val SCWA per Depth (Test SCWA highlighted)")
        plt.legend()
        fn = os.path.join(working_dir, f"{ds_name}_best_scwa_summary.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating summary bar plot: {e}")
        plt.close()
