from datasets import load_dataset

# Load four HuggingFace local datasets from CSV files
# The datasets correspond to four benchmarks: SFRFG, IJSJF, GURSG, TSHUY.
# Each dataset directory contains train.csv, dev.csv, and test.csv.

# Load SFRFG benchmark dataset
sfrfg_dataset = load_dataset('csv', data_files={
    'train': 'SPR_BENCH/SFRFG/train.csv',
    'dev': 'SPR_BENCH/SFRFG/dev.csv',
    'test': 'SPR_BENCH/SFRFG/test.csv'
}, delimiter=',')

# Load IJSJF benchmark dataset
ijsjf_dataset = load_dataset('csv', data_files={
    'train': 'SPR_BENCH/IJSJF/train.csv',
    'dev': 'SPR_BENCH/IJSJF/dev.csv',
    'test': 'SPR_BENCH/IJSJF/test.csv'
}, delimiter=',')

# Load GURSG benchmark dataset
gursg_dataset = load_dataset('csv', data_files={
    'train': 'SPR_BENCH/GURSG/train.csv',
    'dev': 'SPR_BENCH/GURSG/dev.csv',
    'test': 'SPR_BENCH/GURSG/test.csv'
}, delimiter=',')

# Load TSHUY benchmark dataset
tshuy_dataset = load_dataset('csv', data_files={
    'train': 'SPR_BENCH/TSHUY/train.csv',
    'dev': 'SPR_BENCH/TSHUY/dev.csv',
    'test': 'SPR_BENCH/TSHUY/test.csv'
}, delimiter=',')

# Print basic information for each loaded dataset.
print("SFRFG dataset splits:", {k: len(v) for k, v in sfrfg_dataset.items()})
print("IJSJF dataset splits:", {k: len(v) for k, v in ijsjf_dataset.items()})
print("GURSG dataset splits:", {k: len(v) for k, v in gursg_dataset.items()})
print("TSHUY dataset splits:", {k: len(v) for k, v in tshuy_dataset.items()})