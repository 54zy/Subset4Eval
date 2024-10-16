import torch
import random

def threshold_tensor(tensor):
    return torch.where(tensor > 0.5, torch.tensor(1), torch.tensor(0))

def set_values_to_one(triples, list1, list2):
    for i in range(len(triples)):
        x, y, z = triples[i]
        if x in list1 and y in list2:
            triples[i] = (x, y,1)
    return triples

def random_half_numbers(n, m):
    numbers = list(range(n+1))
    random.shuffle(numbers)
    half = len(numbers) * m
    return numbers[:int(half)]

def adaptest_load(model, path):
    model.load_state_dict(torch.load(path), strict=False)

def adaptest_save(model, path):
    model_dict = model.state_dict()
    model_dict = {k:v for k,v in model_dict.items() if 'theta' in k or 'beta' in k}
    torch.save(model_dict, path)
