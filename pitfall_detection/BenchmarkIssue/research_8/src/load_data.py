from datasets import load_dataset
import random

# Load HuggingFace datasets - Using 4 local datasets as specified
sfrfg = load_dataset('csv', data_files='SPR_BENCH/SFRFG/train.csv')['train']
ijsjf = load_dataset('csv', data_files='SPR_BENCH/IJSJF/train.csv')['train']
gursg = load_dataset('csv', data_files='SPR_BENCH/GURSG/train.csv')['train']
tshuy = load_dataset('csv', data_files='SPR_BENCH/TSHUY/train.csv')['train']

# Function to introduce noise, disturbances to a sequence
def perturb_sequence(sequence):
    sequence = sequence.split()

    # Swap two random tokens if length allows
    if len(sequence) > 1:
        index1, index2 = random.sample(range(len(sequence)), 2)
        sequence[index1], sequence[index2] = sequence[index2], sequence[index1]

    # Random violation - swap some tokens if present
    if random.choice([True, False]):
        random_index = random.randint(0, len(sequence) - 1)
        sequence[random_index] = sequence[random_index].replace('■', '◆') if '■' in sequence[random_index] else sequence[random_index].replace('◆', '■')

    return ' '.join(sequence)

# Transform datasets and apply noise
for dataset in [sfrfg, ijsjf, gursg, tshuy]:
    # Add transformed sequences as new column
    dataset = dataset.map(lambda x: {'perturbed_sequence': perturb_sequence(x['sequence'])})
    # Display some transformed sequences
    for i, data in enumerate(dataset):
        if i < 5:
            print({'original': data['sequence'], 'perturbed': data['perturbed_sequence']})