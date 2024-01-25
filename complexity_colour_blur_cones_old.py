import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import time


import pywt
import py_cwt2d
from matplotlib import pylab as plt
from PIL import Image 
from skimage.color import rgb2gray
from skimage import io, transform



from scipy import stats
from skimage import io, transform

from scipy.signal import argrelextrema

import skimage as ski

import numpy.ma as ma
import csv
from PIL import Image 
from PIL import ImageOps

import os

import os.path
import entropy as ent


import glob

import pandas as pd

import pickle

import math
from random import randint, random

from tifffile import imsave


import matplotlib.pyplot as plt


# Function converting colored picture to the gray scale

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Function doing coarse graining of the image

def coarse_grain(img, depth):

    stack_of_patterns = []  # To keep all the coarse-grained pictures
    stack_of_patterns.append(img)  # Append the original picture
    full_depth = int(np.log2(len(img))) # Estimates the total number of coarse-graining steps that can be done, assuming that we apply 2*2 filter at each step
    for iloop in range(1, depth):

        parental_pattern = stack_of_patterns[-1]  # Takes the last coarse-grained pattern from the previous iteration
        #renormalized_pattern = np.zeros((2**(full_depth-iloop), 2**(full_depth-iloop)))  # Prepare an empty matrix to create a renormalized picture
        renormalized_pattern=ski.filters.gaussian(img, sigma=(4*iloop, 4*iloop), truncate=3.5, channel_axis=-1)
        '''
        for jloop in range(2**(full_depth-iloop)):   # The core averaging procedure
            for kloop in range(2**(full_depth-iloop)):

                pixel = 1/4. * (parental_pattern[kloop*2][jloop*2]+parental_pattern[kloop*2+1][jloop*2]+parental_pattern[kloop*2][jloop*2+1]+parental_pattern[kloop*2+1][jloop*2+1] )
                renormalized_pattern[kloop][jloop] = pixel
        '''
        stack_of_patterns.append(renormalized_pattern) # Append the renormalized picture to the stack

    return stack_of_patterns


# Function increasing resolution of a picture from N*N to 2N*2N - it is needed to compute overlap between subsequent renormalized patterns (they have to be of the same resolution)

def single_blow_up(img): #FIXME fix that first!

    side = len(img)
    new_side = 2*side

    #blown_up = np.zeros((new_side, new_side))
    blown_up = transform.resize(img, (new_side,new_side), anti_aliasing=False)
    '''
    for iloop in range(new_side):
        for jloop in range(new_side):

            blown_up[iloop][jloop] = img[iloop//2][jloop//2]
    '''
    return blown_up

# Function increasing resolution of a picture from N*N to 2^nn N * 2^nn N

def blow_up(img, nn):

    for iloop in range(nn):

        img = single_blow_up(img)

    return img

# The main function computing partial and overall complexities

def compute_complexities(img): 

    partials = []  # this will store partial complexities C1, C2, C3 etc.
    complexities = []   # this will store cumulative complexities like C1, C1+C2, C1+C2+C3... The final cumulative complexity is the overall one.

## The next six lines are simply to make sure that the image has dimension exactly 2^L * 2^L - unfortunately, this algorithm requires your data to be first scaled to this size
                
    depth1 = np.log2(len(img))  # Computes the possible depth of coarse graining using one side of the image
    depth2 = np.log2(len(img[0]))  # Computes the possible depth of coarse graining using the other side of the image

    assert abs( int(depth1) - depth1 ) < 1e-6, "The first side should be a power of 2"
    assert abs( int(depth2) - depth2 ) < 1e-6, "The second side should be a power of 2"

    depth1 = int(depth1)
    depth2 = int(depth2)

    assert int(depth2) == int(depth1), "Sides must be equal"

    stack = coarse_grain(img, 4)#depth1 - 3)  # Does the coarse-graining to depth = (maximal depth - 3)  -  I assume that the last three patterns are too coarse to bear any interesting information
    #blown_up_stack=stack
    '''
    blown_up_stack = []      # Will store all the coarse-grained images upscaled to the original resolution
    blown_up_stack.append(img)   # Append the original image
    '''
    complexity = 0    # Will be the overall complexity
    partial = []

    for iloop in range(1, len(stack)):

        #blown_up_stack.append(blow_up(stack[iloop], iloop))            

        complexity += np.abs(np.einsum('ij,ij', stack[iloop], stack[iloop-1]) - np.einsum('ij,ij', stack[iloop-1], stack[iloop-1]))    # Add overlap(i,i-1) - overlap(i-1,i-1) to the complexity
        partial.append(np.abs(np.einsum('ij,ij', stack[iloop], stack[iloop-1]) - np.einsum('ij,ij', stack[iloop-1], stack[iloop-1])))  # Remember overlap(i,i-1) - overlap(i-1,i-1) as the partial complexity

        complexities.append(complexity)   # Array of cumulative complexities
        partials.append(partial)   # Array of partial complexities

        #Visualization of coarse-grained patterns; commented out for now

        clrmap='jet'
        datalim=(0, 0.006)
        font_size1=18 
        font_size2=12

        #plt.imshow(blow_up(stack[iloop], iloop), cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(blow_up(stack[iloop], iloop)))
        #plt.show()
    return complexity, partial


def debug_cg(im):
	###DEBUG
	plt.clf()
	plt.imshow(im, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(im))
	plt.savefig('0debug_orig.png')
	print("original_printed")
	###
	gray = rgb2gray(im)
	###DEBUG
	plt.clf()
	plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(gray))
	plt.savefig('0debug_gray.png')
	print("greyscale_printed")
	###
	rs = transform.resize(gray, (512,512), anti_aliasing=False) 
	###DEBUG
	plt.clf()
	plt.imshow(rs, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(rs))
	plt.savefig('0debug_rs.png')
	print("resized_printed")
	###
	norm=np.einsum('ij,ij', rs, rs)
	#intensity of image/square root of norm
	rs=rs/np.sqrt(norm)
	
	profiles = []
	stack_of_patterns = []
	stack_of_patterns.append(rs)
	renorm_pattern = rs

	stack = coarse_grain(rs, 6)
	#print(len(stack))
	i=0
	for el in stack:
	     #print(np.shape(el))
	     plt.clf()
	     plt.imshow(el, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(el))
	     plt.savefig('debug_'+str(i)+'.png')
	     i=i+1



def regression(x,y):
	slope, intercept, r, p, std_err = stats.linregress(x, y)
	y1=slope * x + intercept
	n=len(y)
	'''plt.clf()
	plt.scatter(x, y)
	plt.plot(x, y1)
	plt.savefig('regression_frac_test.png') 
	'''
	delta=(y1-y).to_numpy()
	error_sum=0
	for i in range(len(y)):
		error_sum=error_sum+delta[i]**2
	std_pred=np.sqrt(error_sum/len(y))
	#N is the number of pairs of scores
	#print statistics
	#image
	#standard error of estimate sqrt(sum((prediction - real value)^2)/n)

#calculates the number of local maximums
def local_max_n(x):
	x=np.asarray(x)
	t=argrelextrema(x, np.greater)
	return(len(t[0]))

#gaps in spectrum? does it work?
def local_min_n(x):
	x=np.asarray(x)
	t=argrelextrema(x, np.less)
	return(len(t[0]))

def frequency_cutoff(image):
	image = (image - image.min()) / (image.max() - image.min())
	# set up a range of scales in logarithmic spacing between 1 and 256 (image width / 2 seems to work ehre)
	ss = [1, 2,   4,   8,  16,  32,  64, 128, 256]#np.geomspace(1.0,256.0,10)
	# calculate the complex pycwt and the wavelet normalizations
	coeffs, wav_norm = py_cwt2d.cwt_2d(image, ss, 'mexh')
	# plot an image showing the combinations of all the scales
	level=5
	C = 1.0 / (ss[:level] * wav_norm[:level])
	reconstruction = (C * np.real(coeffs[..., :level])).sum(axis=-1)
	#reconstruction = 1.0 - (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())
	return reconstruction

#enthropy of complexity spectrum
#FIXME

#image number, image, total complexity, partial complexities
def image_stats(im_n, im, partial_complexity):
	features=['512-256', '256-128', '128-64', '64-32', '32-16']
	plt.clf()
	x=range(len(partial_complexity))
	fig, axs = plt.subplots(3, 2)
	#print("???")
	axs[0, 0].imshow(im, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(im))
	axs[0, 0].set_title('original image, n='+str(im_n))
	#print("!!!!")
	axs[0, 1].plot(x,partial_complexity, label='partial complexities')
	axs[0, 1].set_title('gaussian blur')
	#print("you are here")
	renormalized_pattern=ski.filters.gaussian(im, sigma=(4, 4), truncate=3.5, channel_axis=-1)
	axs[1, 0].imshow(renormalized_pattern, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(im))
	axs[1, 0].set_title('sigma=4')
	renormalized_pattern=ski.filters.gaussian(im, sigma=(8, 8), truncate=3.5, channel_axis=-1)
	axs[1, 1].imshow(renormalized_pattern, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(im))
	axs[1, 1].set_title('sigma=8')
	renormalized_pattern=ski.filters.gaussian(im, sigma=(12, 12), truncate=3.5, channel_axis=-1)
	axs[2, 0].imshow(renormalized_pattern, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(im))
	axs[2, 0].set_title('sigma=12')
	renormalized_pattern=ski.filters.gaussian(im, sigma=(16, 16), truncate=3.5, channel_axis=-1)
	axs[2, 1].imshow(renormalized_pattern, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(im))
	axs[2, 1].set_title('sigma=16')
	plt.savefig('image_debug/'+str(im_n)+"_blur_debug.png")
	plt.close(fig)

subset_name='suprematism'
dataset_path="/vol/tcm36/akravchenko/image_complexity/Savoias-Dataset/Images/Suprematism/"
ranking_path="/vol/tcm36/akravchenko/image_complexity/Savoias-Dataset/Images/global_ranking/global_ranking_sup.xlsx"

#complexity_colour_blur.py suprematism "../Savoias-Dataset/Images/Suprematism/" "../Savoias-Dataset/Images/global_ranking/global_ranking_sup.xlsx" blur
subset_name=sys.argv[1]
dataset_path=sys.argv[2]
ranking_path=sys.argv[3]
#cg_type=sys.argv[4]
cg_type='blur_cones'
############load sorted files##########################

image_list_jpg=[]
i=0
list_dir=[int(file.split(".")[0]) for file in os.listdir(dataset_path)]
list_dir.sort()
for fname in list_dir:   
	fpath=dataset_path + '/' + str(fname)+".jpg"
	if (os.path.isfile(fpath)==False):
		fpath=dataset_path + '/' + str(fname)+".png"
	img = Image.open(fpath)
	image_list_jpg.append(img)
#############

image_list=[]
for im in image_list_jpg:
	image_list.append(np.array(im))#(ImageOps.grayscale(im)))

#all 3 colours
complexity_list=[]
complexity_list_partial=[]
local_max_list=[]
local_min_list=[]

df = pd.ExcelFile(ranking_path).parse('Sheet1');
mask=np.zeros(len(df))
cnt=0
for im in image_list: 
	if len(im.shape) == 2:#for greyscale images
		rs = transform.resize(im, (512,512), anti_aliasing=False) 
		#rs=frequency_cutoff(rs)
		norm=np.einsum('ij,ij', rs, rs)
		#intensity of image/square root of norm
		rs=rs/np.sqrt(norm)
		x, partial = compute_complexities(rs)   # Compute all the complexities for the image
		#print(x)

		complexity_list.append(x)
		complexity_list_partial.append(partial)
		local_max_list.append(1+local_max_n(partial))
		local_min_list.append(1+local_min_n(partial))
		image_stats(cnt, rs, partial)
	else:
		im_channels=[im[:,:,0], im[:,:,1], im[:,:,2]]#FIXME there's a more elegant way to do this
		im_intensity=[0, 0, 0]
		j=0
		complexity_channels=[]
		complexity_channels_partial=[]
		for im_c in im_channels:
			#im_c = transform.resize(im_c, (512,512), anti_aliasing=False) 
			#im_c=frequency_cutoff(im_c)
			norm=im_c/256
			coeff=np.sum(norm)/(norm.shape[0]*norm.shape[1]) 
			im_intensity[j]=coeff
			rs = transform.resize(im_c, (512,512), anti_aliasing=False) 
			rs=im_c
			norm=np.einsum('ij,ij', rs, rs)
			#intensity of image/square root of norm
			rs=rs/np.sqrt(norm)
			x, partial = compute_complexities(rs)   # Compute all the complexities for the image
			#print(x)

			j=j+1
			complexity_channels.append(x)
			complexity_channels_partial.append(partial)

		cmpl=im_intensity[0]*complexity_channels[0] + im_intensity[1]*complexity_channels[1] + im_intensity[2]*complexity_channels[2]
		cmpl=cmpl
		cmpl_partial=im_intensity[0]*np.asarray(complexity_channels_partial[0]) + im_intensity[1]*np.asarray(complexity_channels_partial[1]) + np.asarray(complexity_channels_partial[2])*complexity_channels[2]
		cmpl_partial=cmpl_partial
		complexity_list.append(cmpl)
		complexity_list_partial.append(cmpl_partial)
		local_max_list.append(1+local_max_n(cmpl_partial))
		local_min_list.append(1+local_min_n(cmpl_partial))
		image_stats(cnt, im, cmpl_partial)
	cnt=cnt+1	
	if (cnt % 10) == 0:
		print(cnt)
		#break


#complexity_list=complexity_list/np.max(complexity_list)
df['gt']=df['gt'].values
df['ms_total']=complexity_list

#subset_name
#cg_type


features=['512-256', '256-128', '128-64', '64-32', '32-16']
df[features]=complexity_list_partial
df['frac']=df['ms_total'].values-df['512-256'].values-df['32-16'].values
df['local_max']=local_max_list
df['local_min']=local_min_list
df['2parts']=df['ms_total']*((df['local_max']+df['local_min'])/2)

df.to_csv('calculated_mssc/'+cg_type+'_'+subset_name+'_complexity.csv', sep='\t')

with open('calculated_mssc/'+cg_type+'_'+subset_name+'_complexity.pickle', 'wb') as handle:
    pickle.dump(df, handle)



x=range(0,df['ms_total'].to_numpy().shape[0])
y1=df['gt'].to_numpy()
y2=df['ms_total'].to_numpy()

#human vs calculated complexity
plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
#plt.ylim(0,0.2)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title(cg_type+' cg, '+subset_name+' total complexity')
plt.legend(loc='lower right')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_complexity_total.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_complexity_total.eps', format='eps')



#regression
x=df['ms_total']
y=df['gt']
slope, intercept, r, p, std_err = stats.linregress(x, y)
y1=slope * x + intercept
plt.clf()
plt.scatter(x, y)
plt.plot(x, y1, color='orange')
plt.title(cg_type+' cg, '+subset_name+' regression (full set).png')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_regression_total.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_regression_total.eps', format='eps')


f = open("mssc_figures/"+cg_type+'_'+subset_name+'_regression_total.log', "w")
ttt=[slope, intercept, r, p, std_err]
print("slope\tintercept\tr\tp\tstd_err", end='\n', file=f)
print(*ttt, sep='\t', end='\n', file=f)
f.close()



#human vs calculated 2-parts complexity
y1=df['gt'].to_numpy()
y2=df['2parts'].to_numpy()
plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
#plt.ylim(0,0.2)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title(cg_type+' cg, '+subset_name+' total complexity')
plt.legend(loc='lower right')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_complexity_total_2parts.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_complexity_total_2parts.eps', format='eps')
#regression
x=df['2parts']
y=df['gt']
slope, intercept, r, p, std_err = stats.linregress(x, y)
y1=slope * x + intercept
plt.clf()
plt.scatter(x, y)
plt.plot(x, y1, color='orange')
plt.title(cg_type+' cg, '+subset_name+' regression (full set).png')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_regression_total_2parts.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_regression_total_2parts.eps', format='eps')
f = open("mssc_figures/"+cg_type+'_'+subset_name+'_regression_total_2parts.log', "w")
ttt=[slope, intercept, r, p, std_err]
print("slope\tintercept\tr\tp\tstd_err", end='\n', file=f)
print(*ttt, sep='\t', end='\n', file=f)
f.close()





#drop outliers
idx_to_drop=[]
fl_o=False
if subset_name=='suprematism':
	idx_to_drop=[17, 18, 19, 20, 27, 48, 50, 52, 64, 53, 13]
	fl_o=True
if subset_name=='art':
	idx_to_drop=[380, 225, 30, 86,88,108, 119, 83]
	fl_o=True
if subset_name=='scenes':
	idx_to_drop=[22]
	fl_o=True
'''
msk=(df['gt']<50) & (df['ms_total']>0.025)
idx_to_drop = df.index[msk]'''

if fl_o:
	df1=df.drop(idx_to_drop)

	y1=df1['gt'].to_numpy()
	y2=df1['frac'].to_numpy()

	#human vs calculated complexity
	plt.clf()
	plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
	plt.ylim(0,0.2)
	plt.xlabel('human ranking')
	plt.ylabel('multi-scale complexity')
	plt.title(cg_type+' cg, '+subset_name+' complexity (no outliers)')
	plt.legend(loc='lower right')
	plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_complexity_filtered.png')
	plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_complexity_filtered.eps', format='eps')

	#regression
	x=df1['ms_total']
	y=df1['gt']
	slope, intercept, r, p, std_err = stats.linregress(x, y)
	y1=slope * x + intercept
	plt.clf()
	plt.scatter(x, y)
	plt.plot(x, y1, color='orange')
	plt.title(cg_type+' cg, '+subset_name+' regression (no outliers).png')
	plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_regression_filtered.png')
	plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_regression_filtered.eps', format='eps')


	f = open("mssc_figures/"+cg_type+'_'+subset_name+'_regression_filtered.log', "w")
	ttt=[slope, intercept, r, p, std_err]
	print("slope\tintercept\tr\tp\tstd_err", end='\n', file=f)
	print(*ttt, sep='\t', end='\n', file=f)
	f.close()



msk=df['frac']<=0.0001 #FIXME: this is to account for rounding errors
idx_to_drop = df.index[msk]
df=df.drop(idx_to_drop)


y1=df['gt'].to_numpy()
y2=df['frac'].to_numpy()
#human vs calculated complexity
plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
#plt.ylim(0,0.2)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title(cg_type+' cg, '+subset_name+' complexity (middle scales)')
plt.legend(loc='lower right')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_complexity_frac.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_complexity_frac.eps', format='eps')


#regression
x=df['frac']
y=df['gt']
slope, intercept, r, p, std_err = stats.linregress(x, y)
y1=slope * x + intercept
plt.clf()
plt.scatter(x, y)
plt.plot(x, y1, color='orange')
plt.title(cg_type+' cg, '+subset_name+' regression (middle scales).png')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_regression_frac.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_regression_frac.eps', format='eps')

f = open("mssc_figures/"+cg_type+'_'+subset_name+'_regression_frac.log', "w")
ttt=[slope, intercept, r, p, std_err]
print("slope\tintercept\tr\tp\tstd_err", end='\n', file=f)
print(*ttt, sep='\t', end='\n', file=f)
f.close()

'''
for feature in features:
	y2=df[feature].to_numpy()
	plt.clf()
	plt.ylim=(0,0.1)
	plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
	plt.xlabel('human ranking')
	plt.ylabel('multi-scale complexity')
	plt.title('paintings complexity, grayscale, '+feature)
	plt.legend(loc='lower right')
	plt.savefig('sup_complexity_greyscale_'+feature+'.png')
'''





