# array converter
from datasets import load_dataset
import json, sys, os

dataset = load_dataset("daily_dialog")

max_lines = 100
min_len = 5
dataPath = '/notebooks/Inferno_Hex/data/'
file_name="data.tsv"

lines = 0

os.remove(dataPath + file_name)
with open(dataPath + file_name, "a") as f:
    f.write("p1 \t p2\n")
    for dialogs in dataset['test']:
        for i in range(0, len(dialogs["dialog"]), 2):
            try:
                if("\t" in dialogs["dialog"][i] or "\t" in dialogs["dialog"][i+1]):
                    print('tab found!!!')
                pair = dialogs["dialog"][i].strip() + "\t" + dialogs["dialog"][i + 1].strip() + "\n"
                pair = f'{dialogs["dialog"][i].strip()}\t{dialogs["dialog"][i + 1].strip()}\n'
                # print(i, pair)
                f.write(pair)
                lines += 1
                if(lines >= max_lines):
                    sys.exit(0)
            except(IndexError):
                pass
