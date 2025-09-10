from datasets import load_dataset

# Load the datasets from HuggingFace
train_sfrfg = load_dataset('csv', data_files='SPR_BENCH/SFRFG/train.csv', split='train')
dev_sfrfg = load_dataset('csv', data_files='SPR_BENCH/SFRFG/dev.csv', split='train')
test_sfrfg = load_dataset('csv', data_files='SPR_BENCH/SFRFG/test.csv', split='train')

train_ijsjf = load_dataset('csv', data_files='SPR_BENCH/IJSJF/train.csv', split='train')
dev_ijsjf = load_dataset('csv', data_files='SPR_BENCH/IJSJF/dev.csv', split='train')
test_ijsjf = load_dataset('csv', data_files='SPR_BENCH/IJSJF/test.csv', split='train')

train_gursg = load_dataset('csv', data_files='SPR_BENCH/GURSG/train.csv', split='train')
dev_gursg = load_dataset('csv', data_files='SPR_BENCH/GURSG/dev.csv', split='train')
test_gursg = load_dataset('csv', data_files='SPR_BENCH/GURSG/test.csv', split='train')

train_tshuy = load_dataset('csv', data_files='SPR_BENCH/TSHUY/train.csv', split='train')
dev_tshuy = load_dataset('csv', data_files='SPR_BENCH/TSHUY/dev.csv', split='train')
test_tshuy = load_dataset('csv', data_files='SPR_BENCH/TSHUY/test.csv', split='train')

# Print out the datasets
print("Loaded SFRFG dataset:", train_sfrfg, dev_sfrfg, test_sfrfg)
print("Loaded IJSJF dataset:", train_ijsjf, dev_ijsjf, test_ijsjf)
print("Loaded GURSG dataset:", train_gursg, dev_gursg, test_gursg)
print("Loaded TSHUY dataset:", train_tshuy, dev_tshuy, test_tshuy)