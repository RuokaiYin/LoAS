import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib
import numpy as np

### PIE CHART TEMPLATE
# names = ['0','1', '2', '3', '4']
power_ratio_tppe = [0.0129489953356026, 0.04316331778534199, 0.5181340253319003, 0.1140231473742966, 0.3117305141728586]
matplotlib.rcParams['font.size'] = 20

colors = ['#3CB3AD', '#B7C3F3', '#DD7596','#888888','#933C3E']
plt.pie(power_ratio_tppe,  labeldistance=1.1, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' }, colors=colors);
# plt.title('Layer 0',fontsize=18)
# plt.show()
plt.savefig('break_up_tppe.pdf', bbox_inches='tight', pad_inches=0.1)


### PIE CHART TEMPLATE
# names = ['0','1', '2', '3', '4']
power_ratio_tppe = [23.9,65.9,10.2]
matplotlib.rcParams['font.size'] = 20
#'#FEFAE0'
colors = ['#5F6F52','#B99470','#E48F45']
plt.pie(power_ratio_tppe,  labeldistance=1.1, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' }, colors=colors);
# plt.title('Layer 0',fontsize=18)
# plt.show()
plt.savefig('break_up_system.pdf', bbox_inches='tight', pad_inches=0.1)