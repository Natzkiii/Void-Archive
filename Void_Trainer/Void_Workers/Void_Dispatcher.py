from torch.utils.data import Dataset
import sys
import torch as torch
import re
import csv
import sqlite3





class CustomDataset(Dataset):
    def __init__(self, path: str, tokenizer, dataset_line_max_length):
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.csv'):
                reader = csv.reader(f)
                self.input_data = [(
                    pair[0] + " " + pair[1]) for pair in reader]

            elif path.endswith('.tsv'):
                reader = csv.reader(f, delimiter='\t')
                self.input_data = [(
                    pair[0] + " " + pair[1]) for pair in reader]

            elif path.endswith(".db"):
                con = sqlite3.connect(path)
                cur = con.cursor()
                cur.execute("SELECT content FROM dataset")

                self.input_data = [(content[0])
                                   for content in cur.fetchall()]

            self.X_encoded = tokenizer(self.input_data, max_length=dataset_line_max_length,
                                       truncation=True, padding='longest', return_tensors="pt")
            self.input_ids = self.X_encoded['input_ids']
            self.attention_mask = self.X_encoded['attention_mask']
            self.labels = self.X_encoded['input_ids']

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx], self.labels[idx])