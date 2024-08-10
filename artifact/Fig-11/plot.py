import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


fig, ax = plt.subplots(figsize=(3, 2), dpi=150)
with open("FT_artifact.txt", "r") as text_file:
    lines = text_file.readlines()
    vgg16 = [float(value) for value in lines[0].strip().split(',') if value]
    resnet = [float(value) for value in lines[1].strip().split(',') if value]    
# resnet = [90.5,89.3,90.6,90.7,90.7]
# vgg16 = [91.3,90.6,91.2,91.2,91.2]
labels = ['Origin', 'Mask', 'FT-e1', 'Ft-e5', 'FT-e10']

ax.yaxis.set_minor_locator(AutoMinorLocator(2))
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.5)

colors = ['#44599b']
ax.set_ylabel('Accuracy (%)',fontsize=15)

ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)

ax.scatter(labels, resnet, s=80, c='orange', alpha=0.85,zorder=4,edgecolor='black',linewidth=0.8)
ax.scatter(labels, vgg16, s=80, c='#4793AF', alpha=0.85,zorder=4,edgecolor='black',linewidth=0.8)
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)
plt.savefig('ft_accuracy.pdf', bbox_inches='tight',pad_inches=0.1)