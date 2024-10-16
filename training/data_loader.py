import pandas as pd
import json
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def load_data(dataset):
    triplets = pd.read_csv(f'../dataset/{dataset}/triples.csv', encoding='utf-8').to_records(index=False)
    metadata = json.load(open(f'../dataset/{dataset}/metadata.json', 'r'))
    concept_map = json.load(open(f'../data/{dataset}/concept_map.json', 'r'))
    concept_map = {int(k):v for k,v in concept_map.items()}
    return triplets, metadata

def prepare_loaders(triplets, batch_size):
    train_triplets, val_triplets = train_test_split(triplets, test_size=0.2, random_state=42)
    train_triplets = [(int(x), int(y), float(z)) for x, y, z in train_triplets]
    val_triplets = [(int(x), int(y), float(z)) for x, y, z in val_triplets]

    train_loader = DataLoader(train_triplets, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_triplets, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
