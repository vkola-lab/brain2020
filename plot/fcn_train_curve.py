import matplotlib 
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
import matplotlib.pyplot as plt
import numpy as np

log_file = '../log/fcn_train.txt'
Lists = []

with open(log_file, 'r') as f:
    List = []
    for line in f:
        if 'th epoch validation confusion matrix' not in line:
            if List: Lists.append(List)
            List = []
        else:
            List.append(float(line.strip('\n')[-6:]))

Lists = np.array(Lists)
print(Lists.shape)
groups, ave, std = [], [], []

g = 5

for i in range(150//g):
    group = Lists[:, i*g:i*g+g].reshape(g*5, -1)
    groups.append(group)
    ave.append(np.mean(group))
    std.append(np.std(group))

x = [i*20*g for i in range(len(ave))]

# plt.errorbar(x, ave, yerr=std, marker='s', mfc='red', fmt='', capsize=2)
plt.xlabel('Number of patches', fontsize=12, weight='bold')
plt.ylabel('Average validation accuracy', fontsize=12, weight='bold')
plt.boxplot(groups)
plt.savefig('fcn_train_plot.tif', dpi=300)
plt.show()

        