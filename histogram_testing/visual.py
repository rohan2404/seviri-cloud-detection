import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# FOR CHANNEL 2 WE WANT THE FIRST GAUSSIAN
# FOR CHANNEL 9 WE WANT THE LAST GAUSSIAN

data1 = np.array(pd.read_csv("histogram_testing/histogram_ch2_z12_x300_y800_box100.csv")["channel"])
data2 = np.array(pd.read_csv("histogram_testing/histogram_ch9_z12_x300_y800_box100.csv")["channel"])
data3 = np.array(pd.read_csv("histogram_testing/histogram_ch2_z12_x300_y400_box100.csv")["channel"])
data4 = np.array(pd.read_csv("histogram_testing/histogram_ch9_z12_x300_y400_box100.csv")["channel"])
data5 = np.array(pd.read_csv("histogram_testing/histogram_ch2_z12_x600_y400_box100.csv")["channel"])
data6 = np.array(pd.read_csv("histogram_testing/histogram_ch9_z12_x600_y400_box100.csv")["channel"])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Histograms (100 bins)", fontsize=16)

datasets = [
    (data1, "Channel 2 | x=300, y=800"),
    (data2, "Channel 9 | x=300, y=800"),
    (data3, "Channel 2 | x=300, y=400"),
    (data4, "Channel 9 | x=300, y=400"),
    (data5, "Channel 2 | x=600, y=400"),
    (data6, "Channel 9 | x=600, y=400"),
]

for ax, (data, title) in zip(axes.flatten(), datasets):
    ax.hist(data, bins=100)
    ax.set_title(title)
    ax.set_xlabel("Channel Value")
    ax.set_ylabel("Frequency")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()