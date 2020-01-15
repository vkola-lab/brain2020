from heatmap_plot import plot_heatmap, plot_complete_heatmap
from riskmap_plot import plot_riskmap, plot_complete_riskmap
import cv2
import numpy as np
import subprocess
import matplotlib.pyplot as plt

plot_heatmap('../DPMs/fcn_exp1/', figsize=(9, 4))
plot_complete_heatmap('../DPMs/fcn_exp1/', figsize=(3, 2))

plot_riskmap('../DPMs/fcn_exp1/', figsize=(9, 4))
plot_complete_riskmap('../DPMs/fcn_exp1/', figsize=(3, 2))

keyword = 'heat'
imga = cv2.imread('./{}map.tif'.format(keyword))[:1100, :, :]
imgb = cv2.imread('./supple_{}map_axial.tif'.format(keyword))
imgc = cv2.imread('./supple_{}map_coronal.tif'.format(keyword))
imgd = cv2.imread('./supple_{}map_sagittal.tif'.format(keyword))
print(imga.shape, imgb.shape, imgc.shape, imgd.shape)

row2 = np.concatenate((imgb, imgc, imgd), axis=1)
print(row2.shape)

whole = np.concatenate((imga, row2), axis=0)
print(whole.shape)
a = 130
b = 120
whole[-600:, a:a+900, :] = imgb
whole[-600:, 1000-b:1900-b, :] = imgc
whole[-600:, 1700:2600, :] = imgd
whole[-600:, 2600:, :] = 255

cv2.imwrite('fig3.tif', whole)


keyword = 'risk'
imga = cv2.imread('./{}map.tif'.format(keyword))[:1050, :, :]
imgb = cv2.imread('./supple_{}map_axial.tif'.format(keyword))
imgc = cv2.imread('./supple_{}map_coronal.tif'.format(keyword))
imgd = cv2.imread('./supple_{}map_sagittal.tif'.format(keyword))
print(imga.shape, imgb.shape, imgc.shape, imgd.shape)

row2 = np.concatenate((imgb, imgc, imgd), axis=1)
print(row2.shape)

whole = np.concatenate((imga, row2), axis=0)
print(whole.shape)
a = 130
b = 120
whole[-600:, a:a+900, :] = imgb
whole[-600:, 1000-b:1900-b, :] = imgc
whole[-600:, 1700:2600, :] = imgd
whole[-600:, 2600:, :] = 255

cv2.imwrite('fig2.tif', whole)
