import matplotlib.pyplot as plt
import numpy as np

We first load experiment_data.npy and focus on the “batch_size_tuning / SPR_BENCH” results, which contain a dict keyed by batch-size strings.  
For every batch size we extract the arrays stored under losses[train], losses[val] and metrics[val] (SCWA) together with the epoch indices.  
We plot four separate “epoch curve” figures: one training-loss curve and one validation-loss curve combined in a single figure for each batch size (4 figures total).  
The fifth and final figure is a bar chart comparing the held-out test SCWA across the four batch sizes so that we can visually spot the best configuration.  
Each figure is wrapped in an independent try–except block, is titled clearly, saved inside working_dir with a descriptive name, and then closed.  
If the data file is missing or corrupted the script prints an error and exits gracefully.  
After plotting we also print to stdout a compact table of {batch_size : test_SCWA} and highlight (with an asterisk) the best score.