import sys
sys.path.append('../')
from utils import get_AD_risk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import zoom
from heatmap_plot import upsample, resize

def plot_riskmap(path, figsize):
    filenames = ['ADNI_128_S_1409_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070821114304781_S33787_I69400.npy', \
                 'ADNI_062_S_0730_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070424120556863_S17062_I50487.npy', \
                 'ADNI_033_S_0923_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070304125438114_S19544_I42509.npy', \
                 'ADNI_018_S_0055_MR_MPR____N3__Scaled_2_Br_20081008152513256_S16960_I119795.npy']
    risks = []
    MRIs = []
    for filename in filenames:
        risk = upsample(get_AD_risk(np.load(path + filename)))
        mri = np.load('/data/datasets/ADNI_NoBack/'+filename)
        risks.append(risk)
        MRIs.append(mri)

    fig = plt.figure(figsize=figsize, dpi=300)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(3, 8),
                     axes_pad=0.00,
                     aspect = True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.05,
                     )

    font_dict = {'fontweight': 'bold', 'fontsize': 14}
    titlename = ['  AD subject 1  ', '  AD subject 2  ', '  NL subject 1  ', '  NL subject 2  ']

    for i in range(4):
        im = grid[i*2].imshow(MRIs[i][:, :, 40].transpose((1, 0))[::-1, :], cmap='gray', vmin=-1, vmax=2.5)
        grid[i*2].axis('off')
        im = grid[i*2+1].imshow(risks[i][:, :, 40].transpose((1, 0))[::-1, :], cmap='bwr', vmin=0, vmax=1)
        grid[i*2+1].axis('off')
        grid[i*2+1].set_title(titlename[i], fontdict=font_dict, loc='right', color = "k")

    for i in range(4):
        im = grid[i*2+8].imshow(np.rot90(MRIs[i][:, 100, :]), cmap='gray', vmin=-1, vmax=2.5)
        grid[i*2+8].axis('off')
        im = grid[i*2+9].imshow(np.rot90(risks[i][:, 100, :]), cmap='bwr', vmin=0, vmax=1)
        grid[i*2+9].axis('off')

    for i in range(4):
        im = grid[i*2+16].imshow(resize(np.rot90(MRIs[i][48, :, :])), cmap='gray', vmin=-1, vmax=2.5)
        grid[i*2+16].axis('off')
        im = grid[i*2+17].imshow(resize(np.rot90(risks[i][48, :, :])), cmap='bwr', vmin=0, vmax=1)
        grid[i*2+17].axis('off')

    cbar = grid[9].cax.colorbar(im, drawedges=False)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(11)

    fig.savefig('./riskmap.tif', dpi=300)


def plot_complete_riskmap(path, figsize):
    filename = 'ADNI_128_S_1409_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070821114304781_S33787_I69400.npy'
    risk = upsample(get_AD_risk(np.load(path + filename)))
    mri = np.load('/data/datasets/ADNI_NoBack/'+filename)
    font_dict = {'fontweight': 'bold', 'fontsize': 14}

    # axial plot
    fig = plt.figure(figsize=figsize, dpi=300)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(6, 8),
                     axes_pad=0.00,
                     aspect = True,
                    #  cbar_location="right",
                    #  cbar_mode="single",
                    #  cbar_size="5%",
                    #  cbar_pad=0.05,
                     )
    for step in range(3):
        for i in range(8):
            im = grid[step*16+i].imshow(mri[:, :, 7*(i+step*8)].transpose((1, 0))[::-1, :], cmap='gray', vmin=-1, vmax=2.5)
            grid[step*16+i].axis('off')
            im = grid[step*16+i+8].imshow(risk[:, :, 7*(i+step*8)].transpose((1, 0))[::-1, :], cmap='bwr', vmin=0, vmax=1)
            grid[step*16+i+8].axis('off')
    # grid[0].set_title('(b)', fontdict=font_dict, loc='right', color = "k")
    # cbar = grid[9].cax.colorbar(im, drawedges=False)
    # for l in cbar.ax.yaxis.get_ticklabels():
    #     l.set_weight("bold")
    #     l.set_fontsize(11)
    fig.savefig('./supple_riskmap_axial.tif', dpi=300)

    # coronal plot
    fig = plt.figure(figsize=figsize, dpi=300)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(6, 8),
                     axes_pad=0.00,
                     aspect=True,
                    #  cbar_location="right",
                    #  cbar_mode="single",
                    #  cbar_size="5%",
                    #  cbar_pad=0.05,
                     )
    for step in range(3):
        for i in range(8):
            im = grid[step * 16 + i].imshow(np.rot90(mri[:, 15+7*(i+step*8), :]), cmap='gray', vmin=-1, vmax=2.5)
            grid[step * 16 + i].axis('off')
            im = grid[step * 16 + i + 8].imshow(np.rot90(risk[:, 15+7*(i+step*8), :]), cmap='bwr', vmin=0, vmax=1)
            grid[step * 16 + i + 8].axis('off')
    # grid[0].set_title('(c)', fontdict=font_dict, loc='right', color = "k")
    # cbar = grid[9].cax.colorbar(im, drawedges=False)
    # for l in cbar.ax.yaxis.get_ticklabels():
    #     l.set_weight("bold")
    #     l.set_fontsize(11)
    fig.savefig('./supple_riskmap_coronal.tif', dpi=300)

    # sagittal plot
    fig = plt.figure(figsize=figsize, dpi=300)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(6, 8),
                     axes_pad=0.00,
                     aspect=True,
                    #  cbar_location="right",
                    #  cbar_mode="single",
                    #  cbar_size="5%",
                    #  cbar_pad=0.05,
                     )
    for step in range(3):
        for i in range(8):
            im = grid[step * 16 + i].imshow(resize(np.rot90(mri[7 * (i + step * 8), :, :])), cmap='gray', vmin=-1, vmax=2.5)
            grid[step * 16 + i].axis('off')
            im = grid[step * 16 + i + 8].imshow(resize(np.rot90(risk[7 * (i + step * 8), :, :])), cmap='bwr', vmin=0, vmax=1)
            grid[step * 16 + i + 8].axis('off')
    # grid[0].set_title('(d)', fontdict=font_dict, loc='right', color = "k")
    # cbar = grid[9].cax.colorbar(im, drawedges=False)
    # for l in cbar.ax.yaxis.get_ticklabels():
    #     l.set_weight("bold")
    #     l.set_fontsize(11)

    fig.savefig('./supple_riskmap_sagittal.tif', dpi=300)


if __name__ == '__main__':
    plot_riskmap('../DPMs/fcn_exp1/', figsize=(9, 4))
    plot_complete_riskmap('../DPMs/fcn_exp1/', figsize=(3, 2))