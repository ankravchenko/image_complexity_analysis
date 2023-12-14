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

import skimage as ski

import numpy.ma as ma

from PIL import Image 
from PIL import ImageOps

import os
import glob
import pandas as pd
import pickle

import math
from random import randint, random
#from tifffile import imsave
import matplotlib.pyplot as plt


#Converting colored picture to the gray scale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

#Coarse graining with Gaussian blur

def coarse_grain_blur(img, depth):

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

###############################################################


#Coarse graining with wavelets
def coarse_grain(img, depth): #FIXME rename into coarse_grain_wavelets and make choosing into a command line argument
    #wavelet decomposition
    image = (img - img.min()) / (img.max() - img.min()) #FIXME: do we really need this, since we already normalize extensively? for now we use it to match image intensity to reconstruction
    ss = np.geomspace(1.0,256.0,num=50)#set up a range of scales in logarithmic spacing between 1 and 256 
    coeffs, wav_norm = py_cwt2d.cwt_2d(image, ss, 'mexh')
    ######################

    stack_of_patterns = []  # To keep all the coarse-grained pictures
    stack_of_patterns.append(image)  # Append the original picture
    ###############wavelet partial reconstruction 
    for iloop in range(1, depth):
        C = 1.0 / (ss[iloop:] * wav_norm[iloop:])
        reconstruction = (C * np.real(coeffs[..., iloop:])).sum(axis=-1)
        reconstruction = 1.0 - (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())
	
        reconstruction=reconstruction/np.max(reconstruction)#FIXME? do we need it?
        stack_of_patterns.append(reconstruction) #Append the renormalized picture to the stack
    #################normalize images here FIXME window with max value instead of multiple images
    indmax=np.argmax(stack_of_patterns[0])
    valmax=np.max(stack_of_patterns[0])
    for i in range(len(stack_of_patterns)):
        valmax_i=np.max(stack_of_patterns[i])
        stack_of_patterns[i]=stack_of_patterns[i]*valmax/valmax_i
    return stack_of_patterns


#median filter. might not work because images need to be pre-normalized differently. use complexity_rank_colour.py for now #FIXME
def coarse_grain_rank(img, depth):

    stack_of_patterns = []  # To keep all the coarse-grained pictures
    stack_of_patterns.append(img)  # Append the original picture
    cg_steps=[1,2,4,16,64,128]#this is for debug and testing medians other than 2**
    for iloop in range(1, depth):
        renormalized_pattern=median(img, disk(4*iloop))
        rp=renormalized_pattern.astype(float)
        rp=rp/np.max(rp)
        stack_of_patterns.append(rp) # Append the renormalized picture to the stack
    return stack_of_patterns


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

    stack = coarse_grain(img, 50)#depth1 - 3)  # Does the coarse-graining to depth = (maximal depth - 3)  -  I assume that the last three patterns are too coarse to bear any interesting information

    blown_up_stack=stack
    stack=stack[0:20]
    '''
    for level in range(len(stack)):
                stack[level]=normalized=skimage.exposure.equalize_hist(stack[level], nbins=256, mask=None) #histogram equalization, has proven to not help with wavelet reconstruction intensity issur
    '''
    norm=np.einsum('ij,ij', stack[0], stack[0])
    stack=stack/np.sqrt(norm)

    complexity = 0    # Will be the overall complexity
    partial = []

    for iloop in range(1, len(stack)):
       
        complexity += np.abs(np.einsum('ij,ij', stack[iloop], stack[iloop-1]) - np.einsum('ij,ij', stack[iloop-1], stack[iloop-1]))    # Add overlap(i,i-1) - overlap(i-1,i-1) to the complexity
        partial.append(np.abs(np.einsum('ij,ij', stack[iloop], stack[iloop-1]) - np.einsum('ij,ij', stack[iloop-1], stack[iloop-1])))  # Remember overlap(i,i-1) - overlap(i-1,i-1) as the partial complexity

        complexities.append(complexity)   # Array of cumulative complexities
        partials.append(partial)   # Array of partial complexities

    return complexity, partial


#outputs the coarse graining results for a given image
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

#outputs intensity of coarse graining stages for a given image
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


#default
subset_name=suprematism
dataset_path="/vol/tcm36/akravchenko/image_complexity/Savoias-Dataset/Images/Suprematism/"
ranking_path="/vol/tcm36/akravchenko/image_complexity/Savoias-Dataset/Images/global_ranking/global_ranking_sup.xlsx"
cg_type="wavelet"

#FIXME if notempty clause
#complexity_colour_main.py suprematism "../Savoias-Dataset/Images/Suprematism/" "../Savoias-Dataset/Images/global_ranking/global_ranking_sup.xlsx" wavelet
subset_name=sys.argv[1]
dataset_path=sys.argv[2]
ranking_path=sys.argv[3]
cg_type=sys.argv[4]


############load sorted files##########################

image_list_jpg=[]
i=0
list_dir=[int(file.split(".")[0]) for file in os.listdir(dataset_path)]
list_dir.sort()
for fname in list_dir:    
	img = Image.open(dataset_path + '/' + str(fname)+".jpg")
	image_list_jpg.append(img)
image_list=[]
for im in image_list_jpg:
	image_list.append(np.array(im))#(ImageOps.grayscale(im)))
#############


#parse human complexity
df = pd.ExcelFile(ranking_path).parse('Sheet1');
mask=np.zeros(len(df))
cnt=0


#calculate complexity
complexity_list=[]
complexity_list_partial=[]

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

		rs = transform.resize(im_c, (512,512)) #FIXME: this denormalizes everything
		#rs=rs/np.max(rs)
		x, partial = compute_complexities(rs)

		#print(x)

		j=j+1
		complexity_channels.append(x)
		complexity_channels_partial.append(partial)

	cmpl=(im_intensity[0]*complexity_channels[0] + im_intensity[1]*complexity_channels[1] + im_intensity[2]*complexity_channels[2])/3
	cmpl_partial=(im_intensity[0]*np.asarray(complexity_channels_partial[0]) + im_intensity[1]*np.asarray(complexity_channels_partial[1]) + im_intensity[2]*np.asarray(complexity_channels_partial[2]))/3
	complexity_list.append(cmpl)
	complexity_list_partial.append(cmpl_partial)


	
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
		#break #this is needed for debug sometimes so that it doesn't calculate complexity for the whole set


#complexity_list=complexity_list/np.max(complexity_list)
df['gt']=df['gt'].values
df['ms_total']=complexity_list


x=range(0,df['ms_total'].to_numpy().shape[0])
y1=df['gt'].to_numpy()
y2=df['ms_total'].to_numpy()

plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title(subset_name+' art complexity')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(subset_name+'_complexity_total_wavelets_20_brightness.png')


all_features=np.geomspace(1.0,256.0,num=50)#[1,2,4,8,16,32,64,128,256]

features=all_features[1:len(complexity_list_partial[0])+1]
df[features]=complexity_list_partial

#df = df.drop(df[df.ms_total>1000].index)

#df1=df.dropna()


#FIXME pickle

df['frac']=df['ms_total'].values-df[1.1198188001321776].values


x=range(0,df['ms_total'].to_numpy().shape[0])
y1=df['gt'].to_numpy()
y2=df['frac'].to_numpy()

plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title(subset_name+' art complexity (fraction)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(subset_name+'_complexity_total_wavelets_20_brightness_frac.png')




df['frac18']=df['ms_total'].values-df[1.1198188001321776].values-df[7.667602301926624].values-df[8.586325209634195].values


y2=df['frac18'].to_numpy()

plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title(subset_name+' complexity (fraction)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(subset_name+'_complexity_total_wavelets_20_brightness_frac18.png')




with open(subset_name+'_wavelet_complexity_brightness.pickle', 'wb') as handle:
    pickle.dump(df, handle)






