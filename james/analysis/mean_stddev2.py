import csv, pickle
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_POINTS = 61

state = pickle.load(open("state.p", "rb"))

linear = [0]*SAMPLE_POINTS
poly = [0]*SAMPLE_POINTS

gcloss = []
glloss = []

gacc = []
bacc = []

cnt = 0
for i, curve in enumerate(state["minima"]):

    with open(f"{curve['path']}/train/version_0/metrics.csv", newline='') as f:
        reader = csv.DictReader(f)
        acc = float(list(reader)[-1]["test_acc"])
        if curve["type"] == "good":
            gacc.append(acc)
        else:
            bacc.append(acc)

gacc = np.array(gacc)
bacc = np.array(bacc)

print(np.mean(gacc), np.std(gacc))
print(np.mean(bacc), np.std(bacc))
