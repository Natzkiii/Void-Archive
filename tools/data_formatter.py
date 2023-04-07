# array converter

dataPath = '/notebooks/Inferno_Hex/data/'

from datasets import load_dataset
import json

dataset = load_dataset("daily_dialog")

max_lines = 3
min_len = 5


lines = 0
with open(dataPath+'daily.json', "w") as f:
    toWrite = []
    for dialogs in dataset['test']:
        for msg in dialogs["dialog"]:
            if(len(msg) <= min_len):
                continue
            lines+=1
            cleanMsg = msg.replace("'",'').replace('"','')
            toWrite.append(cleanMsg)
            if(lines >= max_lines):
                break
        if(lines >= max_lines):
            break
    json.dump(toWrite, f)
