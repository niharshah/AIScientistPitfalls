from datasets import load_dataset

# Load accessible datasets for symbolic sequence tasks
datasets_to_load = ["spr_bench", "boolq", "squad_v2", "yelp_polarity"]

# Load and review each dataset
for dataset_name in datasets_to_load:
    try:
        dataset = load_dataset(dataset_name)
        print(f"\nSuccessfully loaded {dataset_name} dataset:")
        print(dataset)

        # Review dataset features
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                print(f"\n{dataset_name} {split} dataset features:")
                print(dataset[split].features)
    except Exception as e:
        print(f"Unable to load {dataset_name} dataset: {e}")