import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import time

from skimage import io, transform

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

    stack = coarse_grain(img, depth1 - 3)  # Does the coarse-graining to depth = (maximal depth - 3)  -  I assume that the last three patterns are too coarse to bear any interesting information
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
	rs = transform.resize(gray, (512,512)) 
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

# Input file

name = 'Untitled.jpeg'

# Read the file

img = mpimg.imread(name)  
rs = transform.resize(img, (512,512))   
gray = rgb2gray(rs)

'''
img_r=img[:,:,0]
img_g=img[:,:,1]
img_b=img[:,:,2]

total_complexity for each
the sum with coeff

norm=img_r/256
coeff=np.sum(norm)/(norm.shape[0]*norm.shape[1])

'''

#print("Mean =", np.mean(gray), "STD =", np.std(gray))
#gray = gray/np.mean(gray)
#norm = np.einsum('ij,ij', gray, gray)
#gray = gray/np.sqrt(norm)

#print(np.amax(gray), np.amin(gray))
#print(len(gray), len(gray[0]))

# Array to save intermediate results

profiles = []

stack_of_patterns = []
stack_of_patterns.append(gray)
renorm_pattern = gray

stack = coarse_grain(gray, 6)
print(len(stack))

for el in stack:
     print(np.shape(el))
#     plt.imshow(el, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(el))
#     plt.show()

print("you are here")
x = compute_complexities(gray)   # Compute all the complexities for the image
norm=np.einsum('ij,ij', gray, gray)
print(x)
print(norm)
x1=x[0]/norm
print(x1)

cf_all=[]
cs_all=[]
random_all=[]
type0_all=[]

dataset_path="/vol/tcm36/akravchenko/image_complexity/Savoias-Dataset/Images/Scenes/"
ranking_path="/vol/tcm36/akravchenko/image_complexity/Savoias-Dataset/Images/global_ranking/global_ranking_places.xlsx"


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
######FIXME: can be done more optimally########
complexity_list_0=[]
complexity_list_1=[]
complexity_list_2=[]
complexity_list_3=[]
complexity_list_4=[]
###############################################
complexity_list_partial=[]
df = pd.ExcelFile(ranking_path).parse('Sheet1');
mask=np.zeros(len(df))
cnt=0
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

		rs = transform.resize(im_c, (512,512)) 
		norm=np.einsum('ij,ij', rs, rs)
		#intensity of image/square root of norm
		rs=rs/np.sqrt(norm)
		x, partial = compute_complexities(rs)   # Compute all the complexities for the image
		#print(x)

		j=j+1
		complexity_channels.append(x)
		complexity_channels_partial.append(partial)

	cmpl=im_intensity[0]*complexity_channels[0] + im_intensity[1]*complexity_channels[1] + im_intensity[2]*complexity_channels[2]
	cmpl_partial=im_intensity[0]*np.asarray(complexity_channels_partial[0]) + im_intensity[1]*np.asarray(complexity_channels_partial[1]) + np.asarray(complexity_channels_partial[2])*complexity_channels[2]
	complexity_list.append(cmpl)
	complexity_list_partial.append(cmpl_partial)

	cnt=cnt+1	
	if (cnt % 10) == 0:
		print(cnt)
		#break


#complexity_list=complexity_list/np.max(complexity_list)
df['gt']=df['gt'].values
df['ms_total']=complexity_list

with open('scenes_complexity.pickle', 'wb') as handle:
    pickle.dump(df, handle)



x=range(0,df['ms_total'].to_numpy().shape[0])
y1=df['gt'].to_numpy()
y2=df['ms_total'].to_numpy()



#y2=df['frac'].to_numpy()
plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
plt.ylim(0,0.2)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title('scenes complexity')
plt.legend(loc='lower right')
plt.savefig('scenes_complexity_total_blur.png')

features=['512-256', '256-128', '128-64', '64-32', '32-16']
df[features]=complexity_list_partial


df['frac']=df['ms_total'].values-df['512-256'].values-df['32-16'].values


y2=df['frac'].to_numpy()
plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
plt.ylim(0,0.2)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title('scenes complexity (frac)')
plt.legend(loc='lower right')
plt.savefig('scenes_complexity_total_blur_frac.png')


y1_x = ma.masked_array(y1, mask)
y2_x = ma.masked_array(y2, mask)

plt.clf()
plt.scatter(y1_x,y2_x,color='blue', linewidth=2, alpha=0.5)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title('objects complexity, filtered')
plt.legend(loc='lower right')
plt.savefig('objects_complexity_filtered_blur.png')

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





