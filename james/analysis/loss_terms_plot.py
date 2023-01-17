import csv, pickle
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_POINTS = 61

state = pickle.load(open("state.p", "rb"))

linear = [0]*SAMPLE_POINTS
poly = [0]*SAMPLE_POINTS
linear1 = [0]*SAMPLE_POINTS
poly1 = [0]*SAMPLE_POINTS

cnt = 0
for i in [0]:

    cnt += 1

    with open(f"curves2/{i}/curve/version_0/metrics.csv", newline='') as f:
        reader = csv.DictReader(f)
        lines = list(reader)
        for j,row in enumerate(lines[1058:1058+SAMPLE_POINTS]):
            poly[j] += float(row["test_loss"])
            poly1[j] += float(row["test_nll"])

        for j,row in enumerate(lines[1119:1119+SAMPLE_POINTS]): pass
            #poly1[j] += float(row["test_loss"])


for i in range(SAMPLE_POINTS):
    poly[i] /= cnt
    poly1[i] /= cnt


fig, ax = plt.subplots()

ts = np.linspace(0, 1, SAMPLE_POINTS)
ts_mix = []
for t in ts:
    if t < 0.5:
        t *= 153 * 2
    else:
        t -= 0.5
        t = 153+t*163 * 2

    t /= 153+163
    ts_mix.append(t)
ax.plot(ts, poly1, label="Error", linestyle="solid", color="blue")
ax.plot(ts, poly, label="Error + Regularisation", linestyle="solid", color="orange")

#ax.set_yscale('log')
ax.set_xlabel("Curve position")
ax.set_ylabel("Loss")
ax.legend()
plt.show()