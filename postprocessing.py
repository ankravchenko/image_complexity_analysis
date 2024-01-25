import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import time
import pywt
import py_cwt2d

from skimage import io, transform

from skimage.filters.rank import median
from skimage.morphology import disk, ball

from skimage.util import img_as_ubyte

import skimage

import skimage as ski

import numpy.ma as ma

from PIL import Image 
from PIL import ImageOps


from scipy import stats
import os


import glob

import pandas as pd

import pickle

import math
from random import randint, random

from tifffile import imsave

import matplotlib.pyplot as plt

#subset_name='suprematism'
subset_name=sys.argv[1]

blur_path="/vol/tcm36/akravchenko/image_complexity/image_complexity_analysis/calculated_mssc/blur_"+subset_name+"_complexity.csv"
wavelet_path="/vol/tcm36/akravchenko/image_complexity/image_complexity_analysis/calculated_mssc/wavelets_"+subset_name+"_complexity.csv"

df_blur = pd.read_csv(blur_path, delimiter='\t')
df_wavelet = pd.read_csv(wavelet_path, delimiter='\t')


gt=df_blur['gt']
c_blur=df_blur['ms_total'].to_numpy()
c_wavelet=df_wavelet['ms_total'].to_numpy()

c_blur_norm=c_blur/np.max(c_blur)
c_wavelet_norm=c_wavelet/np.max(c_wavelet)
comp=c_blur_norm+c_wavelet_norm
cg_type='sum_wavelets_blur'

plt.clf()
plt.scatter(gt,comp,color='blue', linewidth=2, alpha=0.5)
#plt.ylim(0,0.2)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title(cg_type+' cg, '+subset_name+' total complexity')
plt.legend(loc='lower right')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_complexity_total.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_complexity_total.eps', format='eps')

#regression
x=comp
y=gt
slope, intercept, r, p, std_err = stats.linregress(x, y)
y1=slope * x + intercept
plt.clf()
plt.scatter(x, y)
plt.plot(x, y1, color='orange')
plt.title(cg_type+' cg, '+subset_name+' regression (full set)_lmax.png')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_regression_total_lmax.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_regression_total_lmax.eps', format='eps')
f = open("mssc_figures/"+cg_type+'_'+subset_name+'_regression_total_lmax.log', "w")
ttt=[slope, intercept, r, p, std_err]
print("slope\tintercept\tr\tp\tstd_err", end='\n', file=f)
print(*ttt, sep='\t', end='\n', file=f)
f.close()

