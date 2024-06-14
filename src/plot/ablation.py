import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FixedLocator, AutoMinorLocator

# Data
labels = ['Area', 'Power']
T_2 = [1,1]
T_4 = [1.13,1.08]
T_8 = [1.37,1.25]
X_axis = np.arange(len(labels))

# Create bar chart
fig, ax = plt.subplots(1,1,figsize=(4,1.8),dpi=150)
# bars = ax.bar(labels, values_lif, color='#F4CE14',zorder=4)
# bars = ax.bar(labels, values, bottom =values_lif, color='#A3B763',zorder=4)
# bars = ax.bar(labels, values_spgem, bottom =values, color='#557C55',zorder=4)

plt.bar(X_axis-0.25, T_2, 0.2,color = '#5581B3', label = 'T=4',zorder=3,edgecolor='black',linewidth=0.8)
plt.bar(X_axis, T_4, 0.2,color = '#5581B3', label = 'T=8',zorder=3,edgecolor='black',linewidth=0.8)
plt.bar(X_axis+0.25, T_8, 0.2,color = '#5581B3', label = 'T=16',zorder=3,edgecolor='black',linewidth=0.8)

# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(2))
# plt.grid(which='minor', alpha=0.2)
# plt.grid(which='major', alpha=0.5)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

ax.yaxis.set_minor_locator(AutoMinorLocator(4))
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.5)

# Set title and y-axis label
# ax.set_title('A')
ax.set_ylabel('Normalized Cost', fontsize=14)
ax.set_xticks(X_axis)
plot = ax.set_xticklabels(labels,  fontsize=14)


# ax.set_yscale('log')
ax.set_yticks([1])
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylim(0, 150)
# plt.xticks(rotation=90)
# ax.legend(frameon=False,handletextpad=0.3,fontsize=13)

# Display the figure
# plt.show()
plt.savefig('time_scale_study.pdf', bbox_inches='tight',pad_inches=0.1)