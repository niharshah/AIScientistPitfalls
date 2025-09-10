from datasets import load_dataset

# Loading the datasets from HuggingFace
datasets = {
    'cifar10': load_dataset('cifar10'),
    'mnist': load_dataset('mnist'),
    'svhn_cropped': load_dataset('svhn', 'cropped_digits'),
    'fashion_mnist': load_dataset('fashion_mnist')
}

# Display basic information about the loaded datasets
for name, dataset in datasets.items():
    print(f"{name} Dataset Summary:")
    print(f"Features: {dataset['train'].features}")
    print(f"Sample Entry: {dataset['train'][0]}")
    print("-" * 40)