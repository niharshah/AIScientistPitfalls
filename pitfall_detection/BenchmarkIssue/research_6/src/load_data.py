from datasets import load_dataset

# Load the datasets using paths relevant for local usage
dfwzn_dataset = load_dataset('csv', data_files='SPR_BENCH/DFWZN/train.csv')
jwaeu_dataset = load_dataset('csv', data_files='SPR_BENCH/JWAEU/train.csv')
gursg_dataset = load_dataset('csv', data_files='SPR_BENCH/GURSG/train.csv')
qavbe_dataset = load_dataset('csv', data_files='SPR_BENCH/QAVBE/train.csv')

# Print a preview of the first few entries for each dataset to confirm loading
print("DFWZN Dataset Sample:")
print(dfwzn_dataset['train'].shuffle(seed=42).select(range(5)))

print("\nJWAEU Dataset Sample:")
print(jwaeu_dataset['train'].shuffle(seed=42).select(range(5)))

print("\nGURSG Dataset Sample:")
print(gursg_dataset['train'].shuffle(seed=42).select(range(5)))

print("\nQAVBE Dataset Sample:")
print(qavbe_dataset['train'].shuffle(seed=42).select(range(5)))