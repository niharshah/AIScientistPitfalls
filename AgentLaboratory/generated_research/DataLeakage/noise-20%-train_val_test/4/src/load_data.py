from datasets import load_dataset
from collections import Counter

# Load the SPR_BENCH dataset splits from HuggingFace
train_dataset = load_dataset('SPR_BENCH', split='train')
validation_dataset = load_dataset('SPR_BENCH', split='validation')
test_dataset_1 = load_dataset('SPR_BENCH', split='test[:50%]')
test_dataset_2 = load_dataset('SPR_BENCH', split='test[50%:]')

# Define a function to calculate and print label distribution for a given dataset
def calculate_label_distribution(dataset, dataset_name):
    label_counts = Counter(dataset['label'])
    print(f"\n{dataset_name} Label Distribution:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count / len(dataset):.2%}")

# Calculate label distributions for each split
calculate_label_distribution(train_dataset, "Train Dataset")
calculate_label_distribution(validation_dataset, "Validation Dataset")
calculate_label_distribution(test_dataset_1, "Test Dataset 1")
calculate_label_distribution(test_dataset_2, "Test Dataset 2")