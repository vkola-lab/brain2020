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
ave, std = [], []

for i in range(150//3):
    group = Lists[:, i*3:i*3+3].reshape(15, -1)
    ave.append(np.mean(group))
    std.append(np.std(group))

x = [i*20*3 for i in range(len(ave))]

plt.plot(x, ave)
plt.errorbar(x, ave, yerr=std, marker='s', mfc='red', fmt='', capsize=2)
plt.xlabel('Number of patches', fontsize=12, weight='bold')
plt.ylabel('Average validation accuracy', fontsize=12, weight='bold')
plt.savefig('fcn_train_plot.tif', dpi=300)
plt.show()

        