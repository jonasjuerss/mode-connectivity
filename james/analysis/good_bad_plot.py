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
for i, curve in enumerate(state["paths"]):
    if state["minima"][curve["start"]]["type"] != "good": continue
    if state["minima"][curve["end"]]["type"] != "bad": continue

    cnt += 1

    with open(f"{curve['path']}/linear/version_0/metrics.csv", newline='') as f:
        reader = csv.DictReader(f)
        for j,row in enumerate(list(reader)):
            linear[j] += float(row["test_loss"])

    with open(f"{curve['path']}/curve/version_0/metrics.csv", newline='') as f:
        reader = csv.DictReader(f)
        for j,row in enumerate(list(reader)[502:502+SAMPLE_POINTS]):
            poly[j] += float(row["test_loss"])

    gcloss.append(curve["curve_loss"])
    glloss.append(curve["lin_loss"])

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
for i, curve in enumerate(state["paths"]):
    if state["minima"][curve["start"]]["type"] == "good": continue
    if state["minima"][curve["end"]]["type"] != "bad": continue

    if abs(curve["start"]-curve["end"]) < 2: continue

    cnt += 1

    with open(f"{curve['path']}/linear/version_0/metrics.csv", newline='') as f:
        reader = csv.DictReader(f)
        for j,row in enumerate(list(reader)):
            linear[j] += float(row["test_loss"])

    with open(f"{curve['path']}/curve/version_0/metrics.csv", newline='') as f:
        reader = csv.DictReader(f)
        for j,row in enumerate(list(reader)[441:441+SAMPLE_POINTS]):
            poly[j] += float(row["test_loss"])

    bcloss.append(curve["curve_loss"])
    blloss.append(curve["lin_loss"])

for i in range(SAMPLE_POINTS):
    linear[i] /= cnt
    poly[i] /= cnt

bad_linear = linear[:]
bad_poly = poly[:]



linear = [0]*SAMPLE_POINTS
poly = [0]*SAMPLE_POINTS

bcloss = []
blloss = []

cnt = 0
for i, curve in enumerate(state["paths"]):
    if state["minima"][curve["start"]]["type"] != "good": continue
    if state["minima"][curve["end"]]["type"] != "bad": continue

    cnt += 1

    with open(f"{curve['path']}/linear/version_0/metrics.csv", newline='') as f:
        reader = csv.DictReader(f)
        for j,row in enumerate(list(reader)):
            linear[j] += float(row["test_loss"])

    with open(f"{curve['path']}/curve/version_0/metrics.csv", newline='') as f:
        reader = csv.DictReader(f)
        for j,row in enumerate(list(reader)[441:441+SAMPLE_POINTS]):
            poly[j] += float(row["test_loss"])

    bcloss.append(curve["curve_loss"])
    blloss.append(curve["lin_loss"])

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
ax.plot(ts, good_linear, label="Linear, Good->Good", linestyle="dashed", color="blue")
ax.plot(ts, linear, label="Linear, Good->Bad", linestyle="dashed", color="orange")
ax.plot(ts, bad_linear, label="Linear, Bad->Bad", linestyle="dashed", color="green")
ax.plot(ts, good_poly, label="PolyChain, Good->Good", linestyle="solid", color="blue")
ax.plot(ts_mix, poly, label="PolyChain, Good->Bad", linestyle="solid", color="orange")
ax.plot(ts, bad_poly, label="PolyChain, Bad->Bad", linestyle="solid", color="green")

ax.set_yscale('log')
ax.set_xlabel("Curve position")
ax.set_ylabel("Training loss")
ax.legend()
plt.show()