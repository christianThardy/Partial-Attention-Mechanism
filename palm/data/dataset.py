from datasets import load_dataset
from torch.utils.data import DataLoader, random_split


def load_and_split_dataset(dataset_name, train_ratio=0.8):
    # Load specified dataset using the HuggingFace
    dataset = load_dataset(dataset_name)

    # Calculate number of samples for training set based on the provided train_ratio
    train_size = int(train_ratio * len(dataset['train']))

    # Calculate number of samples for evaluation set
    eval_size = len(dataset['train']) - train_size
    
    # Split dataset into training and evaluation subsets
    return random_split(dataset['train'], [train_size, eval_size])