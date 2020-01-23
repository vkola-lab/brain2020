# -*- coding: utf-8 -*-
"""
Created on Thu May 30 23:46:20 2019
@author: Iluva
"""

import matplotlib.pyplot as plt
import collections


def plot_curve(curve, xs, ys_mean, ys_upper, ys_lower, ax, color, hatch, alpha, line, title, **kwargs):
    assert curve in ['roc', 'pr']
    if curve == 'roc':
        ys_mean = ys_mean[::-1]
        ys_upper = ys_upper[::-1]
        ys_lower = ys_lower[::-1]
        xlabel, ylabel = 'Specificity', 'Sensitivity'
    else:
        xlabel, ylabel = 'Recall', 'Precision'

    p_mean, = ax.plot(
        xs, ys_mean, color=color,
        linestyle=line,
        lw=1.5, alpha=1)

    if hatch:
        p_fill = ax.fill_between(
            xs, ys_lower, ys_upper,
            alpha=alpha,
            facecolor='none',
            edgecolor=color,
            hatch=hatch)
    else:
        p_fill = ax.fill_between(
            xs, ys_lower, ys_upper,
            alpha=alpha,
            color=color)

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.xaxis.set_label_coords(0.5, -0.01)
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.yaxis.set_label_coords(-0.01, 0.5)
    ax.set_title(title, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(ax.get_xticks(), weight='bold')

    ax.set_aspect('equal', 'box')
    ax.set_facecolor('w')
    plt.setp(ax.spines.values(), color='w')
    ax.axhline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    ax.axhline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    ax.axvline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    ax.axvline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    ax.axvline(0.0, linestyle='-', color='k', lw=1, zorder=1)
    ax.axhline(0.0, linestyle='-', color='k', lw=1, zorder=1)

    return p_mean, p_fill


def plot_legend(axes, crv_lgd_hdl, crv_info, neo_lgd_hdl):
    m_name = list(crv_lgd_hdl.keys())
    ds_name = list(crv_lgd_hdl[m_name[0]].keys())

    hdl = collections.defaultdict(list)
    val = collections.defaultdict(list)

    if neo_lgd_hdl:
        for ds in neo_lgd_hdl:
            hdl[ds] += neo_lgd_hdl[ds]
            val[ds] += ['Neurologist', 'Avg. Neurologist']

    convert = {'A':"MRI", 'B':"NoI", 'C':"FUS"}

    for ds in ds_name:
        for m in m_name:
            hdl[ds].append(crv_lgd_hdl[m][ds])
            val[ds].append('{}: {:.3f}$\pm${:.3f}'.format(convert[m], crv_info[m][ds]['auc_mean'], crv_info[m][ds]['auc_std']))

        axes[ds].legend(hdl[ds], val[ds],
                        facecolor='w', prop={"weight":'bold', "size":17},  # frameon=False,
                        bbox_to_anchor=(0.04, 0.04, 0.5, 0.5),
                        loc='lower left')


def plot_neorologist(ax, mode, info):
    assert mode in ['roc', 'pr']

    if mode == 'roc':
        neo_x = [v['specificity'] for k, v in info.items() if k not in ['mean', 'std']]
        neo_y = [v['sensitivity'] for k, v in info.items() if k not in ['mean', 'std']]
        neo_x_mean = info['mean']['specificity']
        neo_y_mean = info['mean']['sensitivity']
        neo_x_std = info['std']['specificity']
        neo_y_std = info['std']['sensitivity']
    else:
        neo_x = [v['sensitivity'] for k, v in info.items() if k not in ['mean', 'std']]
        neo_y = [v['precision'] for k, v in info.items() if k not in ['mean', 'std']]
        neo_x_mean = info['mean']['sensitivity']
        neo_y_mean = info['mean']['precision']
        neo_x_std = info['std']['sensitivity']
        neo_y_std = info['std']['precision']

    p_neo = ax.scatter(neo_x, neo_y, color='r', marker='P', linewidths=1, edgecolors='k', s=7 ** 2, zorder=10)
    p_avg = ax.errorbar(neo_x_mean, neo_y_mean,
                        xerr=neo_x_std, yerr=neo_y_std, fmt='o',
                        markeredgewidth=1, markeredgecolor='k',
                        markerfacecolor='green',
                        markersize=7, marker='P',
                        elinewidth=1.5, ecolor='green',
                        capsize=3, zorder=11)

    return [p_neo, p_avg]
