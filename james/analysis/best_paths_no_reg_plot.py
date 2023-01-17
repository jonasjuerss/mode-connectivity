import csv, pickle
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_POINTS = 61

state = pickle.load(open("state.p", "rb"))

linear = [0]*SAMPLE_POINTS
poly = [0]*SAMPLE_POINTS

gcloss = []
glloss = []

cnt = 0
for i in [0,1,4]:

    cnt += 1

    with open(f"curves2/{i}/linear/version_0/metrics.csv", newline='') as f:
        reader = csv.DictReader(f)
        for j,row in enumerate(list(reader)[:SAMPLE_POINTS]):
            linear[j] += float(row["test_loss"])

    with open(f"curves2/{i}/curve/version_0/metrics.csv", newline='') as f:
        reader = csv.DictReader(f)
        for j,row in enumerate(list(reader)[529:529+SAMPLE_POINTS]):
            poly[j] += float(row["test_loss"])

for i in range(SAMPLE_POINTS):
    linear[i] /= cnt
    poly[i] /= cnt



good_linear = linear[:]
good_poly = poly[:]



linear = [0]*SAMPLE_POINTS
poly = [0]*SAMPLE_POINTS

bcloss = []
blloss = []

cnt = 0
for i in [2,3,7]:

    cnt += 1

    with open(f"curves2/{i}/linear/version_0/metrics.csv", newline='') as f:
        reader = csv.DictReader(f)
        for j,row in enumerate(list(reader)[:SAMPLE_POINTS]):
            linear[j] += float(row["test_loss"])

    with open(f"curves2/{i}/curve/version_0/metrics.csv", newline='') as f:
        reader = csv.DictReader(f)
        for j,row in enumerate(list(reader)[529:529+SAMPLE_POINTS]):
            poly[j] += float(row["test_loss"])

for i in range(SAMPLE_POINTS):
    linear[i] /= cnt
    poly[i] /= cnt


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



ax.plot(ts, good_linear, label="Linear, Good->Best", linestyle="dashed", color="blue")
ax.plot(ts, linear, label="Linear, Bad->Best", linestyle="dashed", color="green")
ax.plot(ts, good_poly, label="PolyChain, Good->Best", linestyle="solid", color="blue")
ax.plot(ts, poly, label="PolyChain, Bad->Best", linestyle="solid", color="green")

ax.set_yscale('log')
ax.set_xlabel("Curve position")
ax.set_ylabel("Training loss")
ax.legend()
plt.show()