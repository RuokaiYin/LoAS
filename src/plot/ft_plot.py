import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


fig, ax = plt.subplots(figsize=(3, 2), dpi=150)

resnet = [90.5,89.3,90.6,90.7,90.7]
vgg16 = [91.3,90.6,91.2,91.2,91.2]
labels = ['Origin', 'PP', 'Epoch1', 'Epoch5', 'Epoch10']

# ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.5)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

colors = ['#44599b']
ax.set_ylabel('Accuracy (%)',fontsize=15)

ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# ax.set_ylim([89, 93])

ax.scatter(labels, resnet, s=80, c='orange', alpha=0.85,zorder=4,edgecolor='black',linewidth=0.8)
ax.scatter(labels, vgg16, s=80, c='#4793AF', alpha=0.85,zorder=4,edgecolor='black',linewidth=0.8)

# plt.show()
plt.savefig('ft_accuracy.pdf', bbox_inches='tight',pad_inches=0.1)