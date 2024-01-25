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

from scipy.signal import argrelextrema
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


# Function converting colored picture to the gray scale

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Function doing coarse graining of the image



def coarse_grain(img, depth):
    #####################wavelet decomposition
    image = img#transform.resize(image, (512,512)) 
    # get an image
    #image = pywt.data.camera()
    image = (image - image.min()) / (image.max() - image.min())
    # set up a range of scales in logarithmic spacing between 1 and 256 (image width / 2 seems to work ehre)
    ss = np.geomspace(1.0,256.0,num=10)#[1,2,4,8,16,32,64,128,256]#np.geomspace(1.0,256.0,num=50)#
    # calculate the complex pycwt and the wavelet normalizations
    coeffs, wav_norm = py_cwt2d.cwt_2d(image, ss, 'mexh')
    ###########################

    stack_of_patterns = []  # To keep all the coarse-grained pictures
    stack_of_patterns.append(image)  # Append the original picture
    full_depth = int(np.log2(len(img))) # Estimates the total number of coarse-graining steps that can be done, assuming that we apply 2*2 filter at each step

    ###############fix image intensity/brightest point here FIXME

    #cg_steps=[1,2,4,8,16,32,64,128,256]#FIXME this is for debug and testing medians other than 2**
    for iloop in range(1, len(ss)):
        #print("cg step "+str(iloop))
        #parental_pattern = stack_of_patterns[-1]  # Takes the last coarse-grained pattern from the previous iteration
        #renormalized_pattern = np.zeros((2**(full_depth-iloop), 2**(full_depth-iloop)))  # Prepare an empty matrix to create a renormalized picture
        #print(str(type(img.dtype)))
        ##############wavelet reconstruction###########
        C = 1.0 / (ss[iloop:] * wav_norm[iloop:])
        reconstruction = (C * np.real(coeffs[..., iloop:])).sum(axis=-1)
        reconstruction = 1.0 - (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())
        ###############################################
        #renormalized_pattern=reconstruction#median(img, disk(4*iloop))#ski.filters.gaussian(img, sigma=(4*iloop, 4*iloop), truncate=3.5, channel_axis=-1)
        #print(str(type(renormalized_pattern.dtype)))
	#print("success")	
        '''
        for jloop in range(2**(full_depth-iloop)):   # The core averaging procedure
            for kloop in range(2**(full_depth-iloop)):

                pixel = 1/4. * (parental_pattern[kloop*2][jloop*2]+parental_pattern[kloop*2+1][jloop*2]+parental_pattern[kloop*2][jloop*2+1]+parental_pattern[kloop*2+1][jloop*2+1] )
                renormalized_pattern[kloop][jloop] = pixel
        '''
	##########FIXME take one window with max value, preserve indices, keep intensity the same (divide/muiltiply the rest transforms to fit)
        #rp=renormalized_pattern.astype(float)
        #rp=rp/np.max(rp)
        reconstruction=reconstruction/np.max(reconstruction)
        stack_of_patterns.append(reconstruction) #Append the renormalized picture to the stack
        #print(str(img))
        #print(str(renormalized_pattern))
    #################normalize images here FIXME
    indmax=np.argmax(stack_of_patterns[0])
    valmax=np.max(stack_of_patterns[0])
    for i in range(len(stack_of_patterns)):
        valmax_i=np.max(stack_of_patterns[i])
        stack_of_patterns[i]=stack_of_patterns[i]*valmax/valmax_i
    return stack_of_patterns


# Function increasing resolution of a picture from N*N to 2N*2N - it is needed to compute overlap between subsequent renormalized patterns (they have to be of the same resolution)

def single_blow_up(img): #FIXME fix that first!

    side = len(img)
    new_side = 2*side

    #blown_up = np.zeros((new_side, new_side))
    blown_up = transform.resize(img, (new_side,new_side), anti_aliasing=True)
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
    #print("started cg")
    stack = coarse_grain(img, 50)#depth1 - 3)  # Does the coarse-graining to depth = (maximal depth - 3)  -  I assume that the last three patterns are too coarse to bear any interesting information
    #print("finished cg")
    #blown_up_stack=stack
    #stack=stack[0:20]
    '''
    for level in range(len(stack)):
                stack[level]=normalized=skimage.exposure.equalize_hist(stack[level], nbins=256, mask=None)
    '''
    norm=np.einsum('ij,ij', stack[0], stack[0])
    stack=stack/np.sqrt(norm)
 
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
	rs = transform.resize(gray, (512,512))
	rs=rs/np.max(rs)
	x = compute_complexities(rs)   # Compute all the complexities for the image
	#norm=np.einsum('ij,ij', rs, rs)
	#print(norm)
	print("total_complexity: " + str(x))

	stack = coarse_grain(rs, 50)
	#print(len(stack))
	i=0
	'''
	for el in stack:
	     #print(np.shape(el))
	     plt.clf()
	     plt.imshow(el, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(el))
	     plt.savefig('debug_'+str(i)+'.png')
	     i=i+1
	'''
	
	N = 10
	fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(15, 15))
	for level in range(len(stack)):
		i = level // N
		j = level % N
		plt.sca(axes[i, j])
		plt.axis('off')
		plt.imshow(stack[level], cmap='gray')
	plt.savefig('wavelets_test.png')
	
	for el in stack:
	     #print(np.shape(el))
	     plt.clf()
	     plt.imshow(el, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(el))
	     plt.savefig('debug_'+str(i)+'.png')
	     i=i+1


def debug_intensity(im):
	###DEBUG
	plt.clf()
	plt.imshow(im, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(im))
	plt.savefig('0debug_orig.png')
	print("original_printed")
	###
	
	gray = rgb2gray(im)
	rs = transform.resize(gray, (512,512))
	rs=rs/np.max(rs)
	
	stack = coarse_grain(rs, 50)
	#for level in range(len(stack)):
		#stack[level]=normalized=skimage.exposure.equalize_hist(sta, nbins=256, mask=None)
	#print(len(stack))
	i=0
	'''
	for el in stack:
	     #print(np.shape(el))
	     plt.clf()
	     plt.imshow(el, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(el))
	     plt.savefig('debug_'+str(i)+'.png')
	     i=i+1

	'''
	N = 10
	fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(15, 15))
	intensity=[]
	for level in range(len(stack)):
		i = level // N
		j = level % N
		plt.sca(axes[i, j])
		plt.axis('off')
		plt.imshow(stack[level], cmap='gray')
		norm=np.einsum('ij,ij', stack[level], stack[level])
		intensity.append(norm)
	plt.savefig('wavelets_test.png')
	plt.clf()
	print(len(intensity))
	x=range(0,len(intensity))
	print(len(x))
	plt.plot(x,intensity, label='self-overlap')
	plt.savefig("image_intensity_wavelet_test.png")
	plt.clf()
	partial_complexity=[]
	for level in range(1,len(stack)):
		norm=np.abs(np.einsum('ij,ij', stack[level], stack[level-1]) - np.einsum('ij,ij', stack[level-1], stack[level-1])) 
		partial_complexity.append(norm)
	x=range(0,len(partial_complexity))
	plt.plot(x,partial_complexity, label='partial complexity at wavelet band')
	plt.savefig("image_pcompl_wavelet_test.png")
	plt.clf()
	'''
	intensity=[]
	#depth1 = np.log2(512)  # Computes the possible depth of coarse graining using one side of the image
	stack = coarse_grain_blur(img, 5)  
	for level in range(len(stack)):
		norm=np.einsum('ij,ij', stack[level], stack[level])
		intensity.append(norm)
	x=range(0,len(intensity))
	print(len(x))
	plt.plot(x,intensity, label='self-overlap')
	plt.savefig("image_intensity_blur_test.png")
	'''	
# Input file


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


#enthropy of complexity spectrum
#FIXME

#image number, image, total complexity, partial complexities
def image_stats(im_n, im, partial_complexity):
	plt.clf()
	x=range(1,len(partial_complexity))
	fig, axs = plt.subplots(2, 2)
	#print("???")
	axs[0, 0].imshow(im, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(im))
	axs[0, 0].set_title('original image, n='+str(im_n))
	#print("!!!!")
	axs[0, 1].plot(x,partial_complexity[1:len(partial_complexity)], label='partial complexities')
	axs[0, 1].set_title('wavelets')
	#print("you are here")
	'''
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
	'''
	plt.savefig('image_debug/'+str(im_n)+"_wavelet_debug.png")
	plt.close(fig)


subset_name='suprematism'
dataset_path="/vol/tcm36/akravchenko/image_complexity/Savoias-Dataset/Images/Suprematism/"
ranking_path="/vol/tcm36/akravchenko/image_complexity/Savoias-Dataset/Images/global_ranking/global_ranking_sup.xlsx"

#complexity_colour_blur.py suprematism "../Savoias-Dataset/Images/Suprematism/" "../Savoias-Dataset/Images/global_ranking/global_ranking_sup.xlsx" blur

subset_name=sys.argv[1]
dataset_path=sys.argv[2]
ranking_path=sys.argv[3]
cg_type=sys.argv[4]
#cg_type='wavelets'

############load sorted files##########################

image_list_jpg=[]
i=0
list_dir=[int(file.split(".")[0]) for file in os.listdir(dataset_path)]
list_dir.sort()
for fname in list_dir:    
	img = Image.open(dataset_path + '/' + str(fname)+".jpg")
	image_list_jpg.append(img)
#############

image_list=[]
for im in image_list_jpg:
	image_list.append(np.array(im))#(ImageOps.grayscale(im)))

#all 3 colours
complexity_list=[]
complexity_list_partial=[]
df = pd.ExcelFile(ranking_path).parse('Sheet1');
mask=np.zeros(len(df))
cnt=0

local_max_list=[]
local_min_list=[]

for im in image_list: 

	im_channels=[im[:,:,0], im[:,:,1], im[:,:,2]]#FIXME there's a more elegant way to do this
	im_intensity=[0, 0, 0]
	j=0
	complexity_channels=[]
	complexity_channels_partial=[]
	for im_c in im_channels:
		norm=im_c/256
		coeff=np.sum(norm)/(norm.shape[0]*norm.shape[1]) 
		im_intensity[j]=coeff

		rs = transform.resize(im_c, (512,512)) #FIXME: something weird is happening here
		#rs=rs/np.max(rs)
		x, partial = compute_complexities(rs)

		#print(x)

		j=j+1
		complexity_channels.append(x)
		complexity_channels_partial.append(partial)

	cmpl=im_intensity[0]*complexity_channels[0] + im_intensity[1]*complexity_channels[1] + im_intensity[2]*complexity_channels[2]
	cmpl_partial=im_intensity[0]*np.asarray(complexity_channels_partial[0]) + im_intensity[1]*np.asarray(complexity_channels_partial[1]) + im_intensity[2]*np.asarray(complexity_channels_partial[2])
	complexity_list.append(cmpl)
	complexity_list_partial.append(cmpl_partial)
	local_max_list.append(1+local_max_n(cmpl_partial))
	local_min_list.append(1+local_min_n(cmpl_partial))
	image_stats(cnt, im, cmpl_partial)


	
	#########greyscale########
	'''
	gray = rgb2gray(im)
	rs = transform.resize(gray, (512,512))
	rs=rs/np.max(rs)
	x, partial = compute_complexities(rs)   # Compute all the complexities for the image
	#norm=np.einsum('ij,ij', rs, rs)
	#print(norm)
	#print("total_complexity: " + str(x))


	complexity_list.append(x)
	complexity_list_partial.append(partial)
	'''
	cnt=cnt+1	
	if (cnt % 10) == 0:
		print(cnt)
		#break


#complexity_list=complexity_list/np.max(complexity_list)
df['gt']=df['gt'].values
df['ms_total']=complexity_list


all_features= np.geomspace(1.0,256.0,num=10)# np.geomspace(1.0,256.0,num=50)#[1,2,4,8,16,32,64,128,256]

features=all_features[1:len(complexity_list_partial[0])+1]
df[features]=complexity_list_partial

df['frac']=df['ms_total'].values-df[features[0]].values-df[features[-1]].values
df['local_max']=local_max_list
df['local_min']=local_min_list
df['2parts']=df['ms_total']*df['local_max']

df.to_csv('calculated_mssc/'+cg_type+'_'+subset_name+'_complexity.csv', sep='\t')

with open('calculated_mssc/'+cg_type+'_'+subset_name+'_complexity.pickle', 'wb') as handle:
    pickle.dump(df, handle)


#df = df.drop(df[df.ms_total>1000].index)

#df1=df.dropna()



x=range(0,df['ms_total'].to_numpy().shape[0])
y1=df['gt'].to_numpy()
y2=df['ms_total'].to_numpy()

#human vs calculated complexity
plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
plt.ylim(0,0.2)
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

#human vs number of peaks
y1=df['gt'].to_numpy()
y2=df['local_max'].to_numpy()
plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
#plt.ylim(0,0.2)
plt.xlabel('human ranking')
plt.ylabel('local maximums')
plt.title(cg_type+' cg, '+subset_name+' total complexity, local maximums number')
plt.legend(loc='lower right')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_complexity_total_lmax.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_complexity_total_lmax.eps', format='eps')
#regression
x=df['local_max']
y=df['gt']
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




y1=df['gt'].to_numpy()
y2=df['frac'].to_numpy()
#human vs calculated complexity
plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
plt.ylim(0,0.2)
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





