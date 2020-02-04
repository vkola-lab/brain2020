from utils_stat import get_roc_info, get_pr_info, load_neurologist_data, calc_neurologist_statistics, read_raw_score
import matplotlib.pyplot as plt
from utils_plot import plot_curve, plot_legend, plot_neorologist
from time import time
from matrix_stat import confusion_matrix, stat_metric
import collections

# collect essentials for plot
roc_info, pr_info = {}, {}
m = 'A'
roc_info[m], pr_info[m] = {}, {}
for ds in ['test', 'AIBL', 'FHS', 'NACC']:
    Scores = []
    for exp_idx in range(5):
        for repe_idx in range(1):
            labels, scores = read_raw_score('../checkpoint_dir/cnn_exp{}/raw_score_{}.txt'.format(exp_idx, ds))
            Scores.append(scores)
    roc_info[m][ds] = get_roc_info(labels, Scores)
    pr_info[m][ds] = get_pr_info(labels, Scores)
m = 'D'
roc_info[m], pr_info[m] = {}, {}
for ds in ['test', 'AIBL', 'FHS', 'NACC']:
    Scores = []
    for exp_idx in range(5):
        for repe_idx in range(3):
            if ds == 'FHS':
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

convert = {'A':"MRI", 'D':"FUS"}


def plot_legend(axes, crv_lgd_hdl, crv_info, neo_lgd_hdl):
    m_name = list(crv_lgd_hdl.keys())
    ds_name = list(crv_lgd_hdl[m_name[0]].keys())

    hdl = collections.defaultdict(list)
    val = collections.defaultdict(list)

    if neo_lgd_hdl:
        for ds in neo_lgd_hdl:
            hdl[ds] += neo_lgd_hdl[ds]
            val[ds] += ['Neurologist', 'Avg. Neurologist']

    convert = {'A':"MRI", 'D':"FUS"}

    for ds in ds_name:
        for m in m_name:
            hdl[ds].append(crv_lgd_hdl[m][ds])
            val[ds].append('{}: {:.3f}$\pm${:.3f}'.format(convert[m], crv_info[m][ds]['auc_mean'], crv_info[m][ds]['auc_std']))

        axes[ds].legend(hdl[ds], val[ds],
                        facecolor='w', prop={"weight":'bold', "size":17},  # frameon=False,
                        bbox_to_anchor=(0.04, 0.04, 0.5, 0.5),
                        loc='lower left')

# roc plot
fig, axes_ = plt.subplots(1, 4, figsize=[24, 6], dpi=100)
axes = dict(zip(['test', 'AIBL', 'FHS', 'NACC'], axes_))
hdl_crv = {'A': {}, 'D': {}}
for i, ds in enumerate(['test', 'AIBL', 'FHS', 'NACC']):
    title = ds if ds != 'test' else "ADNI Test"
    hdl_crv['A'][ds] = plot_curve(curve='roc', **roc_info['A'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': '--',
                                     'title': title})
    hdl_crv['D'][ds] = plot_curve(curve='roc', **roc_info['D'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': None, 'alpha': .2, 'line': '-', 'title': title})
plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=roc_info, neo_lgd_hdl=neo_info)
fig.savefig('./cnn_roc.tif', dpi=300)

# pr plot
fig, axes_ = plt.subplots(1, 4, figsize=[24, 6], dpi=100)
axes = dict(zip(['test', 'AIBL', 'FHS', 'NACC'], axes_))
hdl_crv = {'A': {}, 'D': {}}
for i, ds in enumerate(['test', 'AIBL', 'FHS', 'NACC']):
    title = ds if ds != 'test' else "ADNI Test"
    hdl_crv['A'][ds] = plot_curve(curve='pr', **pr_info['A'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': '//////', 'alpha': .4, 'line': '--',
                                     'title': title})
    hdl_crv['D'][ds] = plot_curve(curve='pr', **pr_info['D'][ds], ax=axes[ds],
                                  **{'color': 'C{}'.format(i), 'hatch': None, 'alpha': .2, 'line': '-', 'title': title})
plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=pr_info, neo_lgd_hdl=neo_info)
fig.savefig('./cnn_pr.tif', dpi=300)


for i, ds in enumerate(['test', 'AIBL', 'FHS', 'NACC']):
    m = 'A'
    Matrix = []
    for exp_idx in range(5):
        labels, scores = read_raw_score('../checkpoint_dir/cnn_exp{}/raw_score_{}.txt'.format(exp_idx, ds))
        Matrix.append(confusion_matrix(labels, scores))
    print('###########################')
    print(m + '--' + ds)
    stat_metric(Matrix)
    print('###########################')

    m = 'D'
    Matrix = []
    for exp_idx in range(5):
        for repe_idx in range(3):
            labels, scores = read_raw_score('../checkpoint_dir/mlp_{}_exp{}/raw_score_{}_{}.txt'.format(m, exp_idx, ds, repe_idx))
            Matrix.append(confusion_matrix(labels, scores))
    print('###########################')
    print(m + '--' + ds)
    stat_metric(Matrix)
    print('###########################')
