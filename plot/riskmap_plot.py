import sys
sys.path.append('../')
from utils import get_AD_risk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import zoom
from heatmap_plot import upsample, resize

def plot_riskmap(path):
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

    fig = plt.figure(dpi=300)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(3, 8),
                     axes_pad=0.00,
                     aspect = True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.05,
                     )

    font_dict = {'fontweight': 'bold', 'fontsize': 10}
    titlename = ['   AD subject', '   AD subject', '   NL subject', '   NL subject']

    for i in range(4):
        im = grid[i*2].imshow(MRIs[i][:, :, 40].transpose((1, 0))[::-1, :], cmap='gray', vmin=-1, vmax=2.5)
        grid[i*2].axis('off')
        grid[i*2].set_title(titlename[i], fontdict=font_dict, loc='left', color = "k")
        im = grid[i*2+1].imshow(risks[i][:, :, 40].transpose((1, 0))[::-1, :], cmap='bwr', vmin=0, vmax=1)
        grid[i*2+1].axis('off')

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


if __name__ == '__main__':
    plot_riskmap('../DPMs/fcn_exp0/')