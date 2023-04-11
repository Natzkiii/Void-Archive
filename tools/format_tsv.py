# array converter

dataPath = '/notebooks/Inferno_Hex/data/'

from datasets import load_dataset
import json, sys

dataset = load_dataset("daily_dialog")

max_lines = 100
min_len = 5


lines = 0
with open(dataPath + 'data.tsv', "a") as f:
    toWrite = []
    for dialogs in dataset['test']:
        for i in range(0, len(dialogs["dialog"]), 2):
            try:
                pair = dialogs["dialog"][i].strip() + "\t" + dialogs["dialog"][i + 1].strip() + "\n"
                # print(i, pair)
                f.write(pair)
                lines += 1
                if(lines >= max_lines):
                    sys.exit(0)
            except(IndexError):
                pass
