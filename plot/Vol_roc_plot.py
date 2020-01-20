from utils_stat import get_roc_info, get_pr_info, calc_neurologist_statistics, read_raw_score
import matplotlib.pyplot as plt
from utils_plot import plot_curve, plot_legend, plot_neorologist
from time import time

# collect essentials for plot
roc_info, pr_info = {}, {}

Labels, Scores = [], []  
for i in range(10):
    labels, scores = read_raw_score('../checkpoint_dir/Vol_RF/raw_score_{}.txt'.format(i))  
    Scores.append(scores)
roc_info = get_roc_info(labels, Scores)
pr_info = get_pr_info(labels, Scores)

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['savefig.facecolor'] = 'w'

# roc plot
fig, axes_ = plt.subplots(1, 2, figsize=[12, 6], dpi=100)
plot_curve(curve='roc', **roc_info, ax=axes_[0],
           **{'color': 'C{}'.format(0), 'hatch': '//////', 'alpha': .4, 'line': '--', 'title': "ROC"})
plot_curve(curve='pr', **pr_info, ax=axes_[1],
           **{'color': 'C{}'.format(1), 'hatch': '//////', 'alpha': .4, 'line': '--', 'title': "PR"})
axes_[0].legend(['AUC: {:.3f} $\pm$ {:.3f}'.format(roc_info['auc_mean'], roc_info['auc_std'])],
                facecolor='w', prop=dict(weight='bold'),  # frameon=False,
                bbox_to_anchor=(0.04, 0.04, 0.5, 0.5),
                loc='lower left')
axes_[1].legend(['AUC: {:.3f} $\pm$ {:.3f}'.format(pr_info['auc_mean'], pr_info['auc_std'])],
                facecolor='w', prop=dict(weight='bold'),  # frameon=False,
                bbox_to_anchor=(0.04, 0.04, 0.5, 0.5),
                loc='lower left')
fig.savefig('./vol_roc.tif', dpi=100)



