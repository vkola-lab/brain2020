from utils_stat import get_roc_info, get_pr_info, load_neurologist_data, calc_neurologist_statistics, read_raw_score
import matplotlib.pyplot as plt
from utils_plot import plot_curve, plot_legend, plot_neorologist
from time import time

# collect essentials for plot
roc_info, pr_info = {}, {}
for m in ['A', 'B', 'C']:
    roc_info[m], pr_info[m] = {}, {}
    for ds in ['test', 'AIBL', 'FHS', 'NACC']:
        Scores = []
        for exp_idx in range(2):
            labels, scores = read_raw_score('../checkpoint_dir/mlp_{}_exp{}/raw_score_{}.txt'.format(m, exp_idx, ds))
            Scores.append(scores)
        roc_info[m][ds] = get_roc_info(labels, Scores)
        pr_info[m][ds] = get_pr_info(labels, Scores)

# neorologist
fn = '../lookupcsv/Ground_Truth_Test.csv'
data = load_neurologist_data(fn)
neo_info = calc_neurologist_statistics(**data)

# plot
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['savefig.facecolor'] = 'w'

# roc plot
fig, axes_ = plt.subplots(1, 4, figsize=[24, 6], dpi=100)
axes = dict(zip(['test', 'AIBL', 'FHS', 'NACC'], axes_))
hdl_crv = {'A': {}, 'B': {}, 'C': {}}
for i, ds in enumerate(['test', 'AIBL', 'FHS', 'NACC']):
    title = ds
    hdl_crv['A'][ds] = plot_curve(curve='roc', **roc_info['A'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': '--',
                                     'title': title})
    hdl_crv['B'][ds] = plot_curve(curve='roc', **roc_info['B'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': '....', 'alpha': .4, 'line': '-.',
                                     'title': title})
    hdl_crv['C'][ds] = plot_curve(curve='roc', **roc_info['C'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': None, 'alpha': .2, 'line': '-', 'title': title})
hdl_neo = plot_neorologist(ax=axes['test'], mode='roc', info=neo_info)
plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=roc_info, neo_lgd_hdl={'test': hdl_neo})
fig.savefig('./roc.tif', dpi=100)

# pr plot
fig, axes_ = plt.subplots(1, 4, figsize=[24, 6], dpi=100)
axes = dict(zip(['test', 'AIBL', 'FHS', 'NACC'], axes_))
hdl_crv = {'A': {}, 'B': {}, 'C': {}}
for i, ds in enumerate(['test', 'AIBL', 'FHS', 'NACC']):
    title = ds
    hdl_crv['A'][ds] = plot_curve(curve='pr', **pr_info['A'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': '--',
                                     'title': title})
    hdl_crv['B'][ds] = plot_curve(curve='pr', **pr_info['B'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': '....', 'alpha': .4, 'line': '-.',
                                     'title': title})
    hdl_crv['C'][ds] = plot_curve(curve='pr', **pr_info['C'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': None, 'alpha': .2, 'line': '-', 'title': title})
hdl_neo = plot_neorologist(ax=axes['test'], mode='pr', info=neo_info)
plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=pr_info, neo_lgd_hdl={'test': hdl_neo})
fig.savefig('./pr.tif', dpi=100)
