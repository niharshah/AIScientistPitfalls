from datasets import load_dataset

# Load the local SPR_BENCH dataset from CSV files
data_files = {
    "train": "./SPR_BENCH/train.csv",
    "dev": "./SPR_BENCH/dev.csv",
    "test": "./SPR_BENCH/test.csv"
}

dataset = load_dataset("csv", data_files=data_files)

# Prepare the data by:
# 1. Splitting the 'sequence' column into a list of tokens.
# 2. Calculating the shape complexity: count of unique shape glyphs (first character of each token).
# 3. Calculating the color complexity: count of unique color letters (if present) from each token.
dataset = dataset.map(lambda x: {
    "tokens": x["sequence"].split(),
    "shape_complexity": len({token[0] for token in x["sequence"].split()}),
    "color_complexity": len({token[1] for token in x["sequence"].split() if len(token) > 1})
})

# Print one example from the train split to verify the processing
print(dataset["train"][0])