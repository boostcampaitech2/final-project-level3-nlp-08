import torch
import numpy as np
import random
import os

import argparse
import pickle

# for reproducibility
def seed_everything(seed: int = 2021):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if device == "cuda:0":
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True

def sort_by_length_with_reference(reference: list, dataset: object):
    
    sorted_dataset = {}
    idx_lst = np.argsort(reference)

    key_lst = dataset.features.keys()

    for key in key_lst:
        sorted_dataset[key] = []

    for key in key_lst:
        temp_lst = sorted_dataset[key]
        original_lst = dataset[key]
        for i in idx_lst:
            temp_lst.append(original_lst[int(i)])

    return sorted_dataset