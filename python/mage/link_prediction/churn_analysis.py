import csv

with open("/home/andi/Memgraph/datasets/churn.csv", newline='') as f:
    spamreader = csv.reader(f, delimiter="'")
    for row in spamreader:
        print(", ".join(row))