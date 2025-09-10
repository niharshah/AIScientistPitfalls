import matplotlib.pyplot as plt
import numpy as np

We load the saved experiment_data.npy, grab the SPR_BENCH section, and for every dropout configuration plot (a) training vs validation loss across epochs and (b) validation accuracy curves. Afterwards we aggregate the final test metrics for every dropout rate and draw a grouped-bar chart comparing acc, cwa, swa and compwa. All plots are wrapped in individual try-except blocks, closed after saving, titled clearly, and stored under working_dir with descriptive filenames.