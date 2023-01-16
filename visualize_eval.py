import numpy as np
import matplotlib.pyplot as plt



files = []
for i in range(1, 10, 2):
    files.append(np.load(f"results/final/fmnist/c{i}-{i+1}/eval/curve.npz"))

disagr_start = np.array([f["disagr_start"] for f in files])
disagr_end = np.array([f["disagr_end"] for f in files])
ts = files[0]["ts"]

def plot_measure(x, ys, color = "b"):
    mean = np.mean(ys, 0)
    std = np.std(ys, 0)
    plt.plot(x,mean, color = color)
    plt.fill_between(x, mean-std, mean+std, alpha = 0.2, color = color)

plot_measure(ts, disagr_start)
plot_measure(np.flip(ts), np.flip(disagr_end, 1), "r")
#plt.show()

te_acc = np.array([f["te_acc"] for f in files])
print(f"system1-2-3: {te_acc[0][0]}, {te_acc[0][-1]}, {te_acc[1][0]}")
print(f"system4-5-6: {te_acc[1][-1]}, {te_acc[2][0]}, {te_acc[2][-1]}")
print(f"system7-8-9: {te_acc[3][0]}, {te_acc[3][-1]}, {te_acc[4][0]}")

te_acc = np.array([f["te_loss"] for f in files])
print(f"system1-2-3: {te_acc[0][0]}, {te_acc[0][-1]}, {te_acc[1][0]}")

te_acc_system1 = 92.885
