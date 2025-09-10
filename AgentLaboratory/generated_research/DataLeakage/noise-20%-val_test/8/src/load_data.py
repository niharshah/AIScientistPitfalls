import pandas as pd
from sklearn.model_selection import train_test_split

# Load the local datasets
train_data = pd.read_csv('SPR_BENCH/train.csv')
dev_data = pd.read_csv('SPR_BENCH/dev.csv')
test_data = pd.read_csv('SPR_BENCH/test.csv')

# Correctly combine datasets using pd.concat instead of append
combined_data = pd.concat([train_data, dev_data, test_data], ignore_index=True)

# Conduct a 70:30 train-test split
train_split, test_split = train_test_split(combined_data, test_size=0.3, random_state=42)

# Display samples of train and test data to verify
print("Train Data Sample:")
print(train_split['sequence'].head())

print("\nTest Data Sample:")
print(test_split['sequence'].head())

# Check representation of predicates: Shape-Count and Color-Position
shape_count_predicate = train_split['sequence'].apply(lambda seq: seq.count('▲') > 5)
color_position_predicate = train_split['sequence'].apply(lambda seq: 'r' in seq)

print("\nShape-Count Predicate Representation (Sample):")
print(shape_count_predicate.head())

print("\nColor-Position Predicate Representation (Sample):")
print(color_position_predicate.head())