from utils_stat import get_roc_info, get_pr_info, load_neurologist_data, calc_neurologist_statistics, read_raw_score
import matplotlib.pyplot as plt
from utils_plot import plot_curve, plot_legend, plot_neorologist
from time import time
from matrix_stat import confusion_matrix, stat_metric
import collections

# collect essentials for plot
roc_info, pr_info = {}, {}
for m in ['A', 'B', 'C']:
    roc_info[m], pr_info[m] = {}, {}
    for ds in ['test', 'AIBL', 'FHS', 'NACC']:
        Scores = []
        for exp_idx in range(5):
            for repe_idx in range(3):
                if ds == 'FHS' and m == 'A':
                    labels, scores = read_raw_score('../checkpoint_dir/mlp_{}_exp{}/raw_score_{}_Full_{}.txt'.format(m, exp_idx, ds, repe_idx))
                else:
                    labels, scores = read_raw_score('../checkpoint_dir/mlp_{}_exp{}/raw_score_{}_{}.txt'.format(m, exp_idx, ds, repe_idx))
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

convert = {'A':"MRI", 'B':"NoI", 'C':"FUS"}

# ADNI roc
fig, axes_ = plt.subplots(1, 2, figsize=[12, 6], dpi=100)
hdl_crv = {'A': {}, 'B': {}, 'C': {}}
ds = 'test'
hdl_crv['A'][ds] = plot_curve(curve='roc', **roc_info['A'][ds], ax=axes_[0],
                                  **{'color': 'C{}'.format(0), 'hatch': '//////', 'alpha': .4, 'line': '--',
                                     'title': "ADNI_Test"})
hdl_crv['B'][ds] = plot_curve(curve='roc', **roc_info['B'][ds], ax=axes_[0],
                                  **{'color': 'C{}'.format(0), 'hatch': '....', 'alpha': .4, 'line': '-.',
                                     'title': "ADNI_Test"})
hdl_crv['C'][ds] = plot_curve(curve='roc', **roc_info['C'][ds], ax=axes_[0],
                                  **{'color': 'C{}'.format(0), 'hatch': None, 'alpha': .2, 'line': '-', 'title': "ADNI_Test"})
hdl_neo = plot_neorologist(ax=axes_[0], mode='roc', info=neo_info)

m_name = list(hdl_crv.keys())
ds_name = list(hdl_crv[m_name[0]].keys())
hdl = collections.defaultdict(list)
val = collections.defaultdict(list)
neo_lgd_hdl={'test': hdl_neo}
for ds in neo_lgd_hdl:
    hdl[ds] += neo_lgd_hdl[ds]
    val[ds] += ['Neurologist', 'Avg. Neurologist']
for ds in ds_name:
    for m in m_name:
        hdl[ds].append(hdl_crv[m][ds])
        val[ds].append('{}: {:.3f}$\pm${:.3f}'.format(convert[m], roc_info[m][ds]['auc_mean'], roc_info[m][ds]['auc_std']))
        axes_[0].legend(hdl[ds], val[ds],
                                facecolor='w', prop={"weight":'bold', "size":17},
                                bbox_to_anchor=(0.04, 0.04, 0.5, 0.5),
                                loc='lower left')


hdl_crv['A'][ds] = plot_curve(curve='pr', **pr_info['A'][ds], ax=axes_[1],
                                  **{'color': 'C{}'.format(0), 'hatch': '//////', 'alpha': .4, 'line': '--',
                                     'title': "ADNI_Test"})
hdl_crv['B'][ds] = plot_curve(curve='pr', **pr_info['B'][ds], ax=axes_[1],
                                  **{'color': 'C{}'.format(0), 'hatch': '....', 'alpha': .4, 'line': '-.',
                                     'title': "ADNI_Test"})
hdl_crv['C'][ds] = plot_curve(curve='pr', **pr_info['C'][ds], ax=axes_[1],
                                  **{'color': 'C{}'.format(0), 'hatch': None, 'alpha': .2, 'line': '-', 'title': "ADNI_Test"})
hdl_neo = plot_neorologist(ax=axes_[1], mode='pr', info=neo_info)

m_name = list(hdl_crv.keys())
ds_name = list(hdl_crv[m_name[0]].keys())
hdl = collections.defaultdict(list)
val = collections.defaultdict(list)
neo_lgd_hdl={'test': hdl_neo}
for ds in neo_lgd_hdl:
    hdl[ds] += neo_lgd_hdl[ds]
    val[ds] += ['Neurologist', 'Avg. Neurologist']
for ds in ds_name:
    for m in m_name:
        hdl[ds].append(hdl_crv[m][ds])
        val[ds].append('{}: {:.3f}$\pm${:.3f}'.format(convert[m], pr_info[m][ds]['auc_mean'], pr_info[m][ds]['auc_std']))
        axes_[1].legend(hdl[ds], val[ds],
                                facecolor='w', prop={"weight":'bold', "size":17},
                                bbox_to_anchor=(0.04, 0.04, 0.5, 0.5),
                                loc='lower left')

fig.savefig('./ADNI_roc_pr.tif', dpi=300)


# roc plot
fig, axes_ = plt.subplots(1, 3, figsize=[18, 6], dpi=100)
axes = dict(zip(['AIBL', 'FHS', 'NACC'], axes_))
hdl_crv = {'A': {}, 'B': {}, 'C': {}}
for i, ds in enumerate(['AIBL', 'FHS', 'NACC']):
    title = ds
    i += 1
    hdl_crv['A'][ds] = plot_curve(curve='roc', **roc_info['A'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': '--',
                                     'title': title})
    hdl_crv['B'][ds] = plot_curve(curve='roc', **roc_info['B'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': '....', 'alpha': .4, 'line': '-.',
                                     'title': title})
    hdl_crv['C'][ds] = plot_curve(curve='roc', **roc_info['C'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': None, 'alpha': .2, 'line': '-', 'title': title})
plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=roc_info, neo_lgd_hdl=None)
fig.savefig('./roc.tif', dpi=300)

# pr plot
fig, axes_ = plt.subplots(1, 3, figsize=[18, 6], dpi=100)
axes = dict(zip(['AIBL', 'FHS', 'NACC'], axes_))
hdl_crv = {'A': {}, 'B': {}, 'C': {}}
for i, ds in enumerate(['AIBL', 'FHS', 'NACC']):
    title = ds
    i += 1
    hdl_crv['A'][ds] = plot_curve(curve='pr', **pr_info['A'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': '--',
                                     'title': title})
    hdl_crv['B'][ds] = plot_curve(curve='pr', **pr_info['B'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': '....', 'alpha': .4, 'line': '-.',
                                     'title': title})
    hdl_crv['C'][ds] = plot_curve(curve='pr', **pr_info['C'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': None, 'alpha': .2, 'line': '-', 'title': title})
plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=pr_info, neo_lgd_hdl=None)
fig.savefig('./pr.tif', dpi=300)


for i, ds in enumerate(['test', 'AIBL', 'FHS', 'NACC']):
    for m in ['A', 'B', 'C']:
        Matrix = []
        for exp_idx in range(5):
            for repe_idx in range(3):
                if ds == 'FHS' and m == 'A':
                    labels, scores = read_raw_score('../checkpoint_dir/mlp_{}_exp{}/raw_score_{}_Full_{}.txt'.format(m, exp_idx, ds, repe_idx))
                else:
                    labels, scores = read_raw_score('../checkpoint_dir/mlp_{}_exp{}/raw_score_{}_{}.txt'.format(m, exp_idx, ds, repe_idx))
                Matrix.append(confusion_matrix(labels, scores))
        print('###########################')
        print(m + '--' + ds)
        stat_metric(Matrix)
        print('###########################')


