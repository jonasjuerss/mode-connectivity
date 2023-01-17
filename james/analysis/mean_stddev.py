import csv, pickle
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_POINTS = 61

state = pickle.load(open("a/state.p", "rb"))

linear = [0]*SAMPLE_POINTS
poly = [0]*SAMPLE_POINTS

gcloss = []
glloss = []

cnt = 0
for i, curve in enumerate(state["paths"]):
    if state["minima"][curve["start"]]["type"] != "good": continue
    if state["minima"][curve["end"]]["type"] != "good": continue

    gcloss.append(curve["curve_loss"])
    glloss.append(curve["lin_loss"])



good_linear = linear[:]
good_poly = poly[:]



linear = [0]*SAMPLE_POINTS
poly = [0]*SAMPLE_POINTS

bcloss = []
blloss = []

cnt = 0
for i, curve in enumerate(state["paths"]):
    if state["minima"][curve["start"]]["type"] == state["minima"][curve["end"]]["type"]: continue

    bcloss.append(curve["curve_loss"])
    blloss.append(curve["lin_loss"])

bbcloss = []
bblloss = []

cnt = 0
for i, curve in enumerate(state["paths"]):
    if state["minima"][curve["start"]]["type"] == "good": continue
    if state["minima"][curve["end"]]["type"] == "good": continue

    if abs(curve["start"]-curve["end"]) < 2: continue

    bbcloss.append(curve["curve_loss"])
    bblloss.append(curve["lin_loss"])





state = pickle.load(open("b/state.p", "rb"))


cnt = 0
for i, curve in enumerate(state["paths"]):
    if state["minima"][curve["start"]]["type"] != "good": continue
    if state["minima"][curve["end"]]["type"] != "good": continue

    gcloss.append(curve["curve_loss"])
    glloss.append(curve["lin_loss"])


cnt = 0
for i, curve in enumerate(state["paths"]):
    if state["minima"][curve["start"]]["type"] == state["minima"][curve["end"]]["type"]: continue

    bcloss.append(curve["curve_loss"])
    blloss.append(curve["lin_loss"])

cnt = 0
for i, curve in enumerate(state["paths"]):
    if state["minima"][curve["start"]]["type"] == "good": continue
    if state["minima"][curve["end"]]["type"] == "good": continue

    if abs(curve["start"]-curve["end"]) < 2: continue

    bbcloss.append(curve["curve_loss"])
    bblloss.append(curve["lin_loss"])




for l, d in [("gc", gcloss), ("gl", glloss), ("bc", bcloss), ("bl", blloss), ("bbc", bbcloss), ("bbl", bblloss)]:
    print(l)
    print(np.mean(d))
    print(np.std(d))
    print(len(d))