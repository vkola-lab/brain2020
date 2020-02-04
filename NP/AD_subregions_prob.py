import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from glob import glob

def resize(plane):
    x, y, z = plane.shape
    return zoom(plane, (181.0/x, 217.0/y, 181.0/z))

def get_AD_risk(raw):
    x1, x2 = raw[0, :, :, :], raw[1, :, :, :]
    risk = np.exp(x2) / (np.exp(x1) + np.exp(x2))
    return risk

def get_AD_prob(seg):
    l1, l2, l3, l4, l5, l6, l7 = [], [], [], [], [], [], []
    for i in range(181):
        for j in range(217):
            for k in range(181):
                if seg[i, j, k] == 15:
                    l1.append(risk[i, j, k])
                if seg[i, j, k] in [6, 13, 25, 28, 29]:
                    l2.append(risk[i, j, k])
                if seg[i, j, k] in [31, 16]:
                    l3.append(risk[i, j, k])
                if seg[i, j, k] in [12]:
                    l4.append(risk[i, j, k])
                if seg[i, j, k] in [5, 18]:
                    l5.append(risk[i, j, k])
                if seg[i, j, k] in [10]:
                    l6.append(risk[i, j, k])
                if seg[i, j, k] in [17, 23]:
                    l7.append(risk[i, j, k])
    ans = [sum(l1) / len(l1), sum(l2) / len(l2), sum(l3) / len(l3), sum(l4) / len(l4), sum(l5) / len(l5),
           sum(l6) / len(l6), sum(l7) / len(l7)]
    return ans

fileList = ['0-1477', '0-1850', '0-1890', '0-2646', '0-2851', '0-3492', '0-3542',
            '0-5119', '0-5268', '1-2127', '1-6072']

for exp_idx in [1, 2, 3, 4]:
    print('{}'.format(exp_idx) * 20)
    compList = glob('../DPMs/fcn_exp{}/*.npy'.format(exp_idx))

    def find(filename, compList):
        for file in compList:
            if filename in file:
                return file.split('/')[-1]
        print('not found ', filename)

    for idx, filename in enumerate(fileList):
        compfilename = find(filename, compList)
        path = './segmen/subject_{}/segments/'.format(idx+1)
        seg_left = nib.load(path + 'lh_combined.mgz').get_data()
        seg_right = nib.load(path + 'rh_combined.mgz').get_data()
        mri = np.load('/data/datasets/FHS_NoBack/{}'.format(compfilename))
        risk = resize(get_AD_risk(np.load('../DPMs/fcn_exp{}/{}'.format(exp_idx, compfilename))))
        print(seg_right.shape, seg_left.shape, mri.shape, risk.shape)
        print(filename + 'left')
        print(get_AD_prob(seg_left))
        print(filename + 'right')
        print(get_AD_prob(seg_right))
        print('############################################################')
    print('{}'.format(exp_idx) * 20)
