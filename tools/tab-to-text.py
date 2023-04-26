dataset = "./data.tsv"
newdataset = "./data.txt"

with open(dataset) as f, open(newdataset, "a+") as ff:
    for line in f:
        for msg in line.strip().split("\t"):
            ff.write(msg + "\n")
