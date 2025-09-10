from datasets import load_dataset

# Load the datasets from HuggingFace
idwep_dataset = load_dataset('SPR_BENCH/IDWEP')
tezgr_dataset = load_dataset('SPR_BENCH/TEZGR')
lyges_dataset = load_dataset('SPR_BENCH/LYGES')
gursg_dataset = load_dataset('SPR_BENCH/GURSG')

# Print sample data from the training set of each dataset
datasets = {
    "IDWEP": idwep_dataset,
    "TEZGR": tezgr_dataset,
    "LYGES": lyges_dataset,
    "GURSG": gursg_dataset
}

for name, dataset in datasets.items():
    print(f"Sample from {name} Training Data:")
    print(dataset['train'][:5])