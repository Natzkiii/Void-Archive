from torch.utils.data import Dataset
from Setting import SETTINGS
import torch as torch
import re
import csv
import pandas as pd

def preprocess(s):
    s = s.replace('\n', '')
    s = s.replace('...', '.')
    s = re.sub(r"([.!?])", r" \1", s)
    s = s.replace('\'', '')
    s = s.replace(',', '')
    s = s.replace('-', ' ')
    s = re.sub(' +', ' ', s)
    return s

class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):
        pairs = []
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.csv'):
                reader = csv.reader(f)
            elif path.endswith('.tsv'):
                reader = csv.reader(f, delimiter='\t')
            else:
                raise ValueError("File type not supported. Only CSV and TSV are supported.")

            for row in reader:
                if len(row) == 2:
                        pairs.append([row[0], row[1]])
                else:
                     raise ValueError("Each row in the dataset should have two columns or two tabs.")

        #preprocess pairs
        for pair in pairs:
            pair[0] = preprocess(pair[0])
            pair[1] = preprocess(pair[1])

        self.input_data = [pair[0] + pair[1] for pair in pairs]
        self.input_data = self.input_data[:50000000]

        self.X_encoded = tokenizer(self.input_data, max_length=512, truncation=True, padding='longest', return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])