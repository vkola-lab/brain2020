import sys
sys.path.append('../')
from utils import get_AD_risk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import zoom
from heatmap_plot import upsample, resize

def plot_supple_riskmap(path):
    filename = 'ADNI_128_S_1409_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070821114304781_S33787_I69400.npy'
    risk = upsample(get_AD_risk(np.load(path + filename)))
    mri = np.load('/data/datasets/ADNI_NoBack/'+filename)

    # axial plot
    fig = plt.figure(dpi=300)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(6, 8),
                     axes_pad=0.00,
                     aspect = True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.05,
                     )
    for step in range(3):
        for i in range(8):
            im = grid[step*16+i].imshow(mri[:, :, 7*(i+step*8)].transpose((1, 0))[::-1, :], cmap='gray', vmin=-1, vmax=2.5)
            grid[step*16+i].axis('off')
            im = grid[step*16+i+8].imshow(risk[:, :, 7*(i+step*8)].transpose((1, 0))[::-1, :], cmap='bwr', vmin=0, vmax=1)
            grid[step*16+i+8].axis('off')
    cbar = grid[9].cax.colorbar(im, drawedges=False)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(11)
    fig.savefig('./supple_riskmap_axial.tif', dpi=300)

    # coronal plot
    fig = plt.figure(dpi=300)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(6, 8),
                     axes_pad=0.00,
                     aspect=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.05,
                     )
    for step in range(3):
        for i in range(8):
            im = grid[step * 16 + i].imshow(np.rot90(mri[:, 15+7*(i+step*8), :]), cmap='gray', vmin=-1, vmax=2.5)
            grid[step * 16 + i].axis('off')
            im = grid[step * 16 + i + 8].imshow(np.rot90(risk[:, 15+7*(i+step*8), :]), cmap='bwr', vmin=0, vmax=1)
            grid[step * 16 + i + 8].axis('off')
    cbar = grid[9].cax.colorbar(im, drawedges=False)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(11)
    fig.savefig('./supple_riskmap_coronal.tif', dpi=300)

    # sagittal plot
    fig = plt.figure(dpi=300)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(6, 8),
                     axes_pad=0.00,
                     aspect=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.05,
                     )
    for step in range(3):
        for i in range(8):
            im = grid[step * 16 + i].imshow(resize(np.rot90(mri[7 * (i + step * 8), :, :])), cmap='gray', vmin=-1, vmax=2.5)
            grid[step * 16 + i].axis('off')
            im = grid[step * 16 + i + 8].imshow(resize(np.rot90(risk[7 * (i + step * 8), :, :])), cmap='bwr', vmin=0, vmax=1)
            grid[step * 16 + i + 8].axis('off')

    cbar = grid[9].cax.colorbar(im, drawedges=False)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(11)

    fig.savefig('./supple_riskmap_sagittal.tif', dpi=300)


if __name__ == '__main__':
    plot_supple_riskmap('../DPMs/fcn_exp0/')