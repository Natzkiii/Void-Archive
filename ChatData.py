from torch.utils.data import Dataset
from Setting import SETTINGS
import re
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
        with open(SETTINGS['Dataset_path'], 'r') as f:
            for line in f:
                pair = line.strip().split('\t')
                pairs.append([pair[0], pair[1]])

        #preprocess pairs
        for pair in pairs:
            pair[0] = preprocess(pair[0].lower())
            pair[1] = preprocess(pair[1].lower())

        self.input_data = ['<sos> ' + pair[0] + ' <eos> ' + '<sos> ' + pair[1] + ' <eos>' for pair in pairs]
        self.input_data = self.input_data[:5000]

        self.X_encoded = tokenizer(self.input_data, max_length=208, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])
