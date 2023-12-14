import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import time

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
    cg_steps=[1,2,4,16,64,128]#FIXME this is for debug and testing medians other than 2**
    for iloop in range(1, depth):
        #print("cg step "+str(iloop))
        #parental_pattern = stack_of_patterns[-1]  # Takes the last coarse-grained pattern from the previous iteration
        #renormalized_pattern = np.zeros((2**(full_depth-iloop), 2**(full_depth-iloop)))  # Prepare an empty matrix to create a renormalized picture
        #print(str(type(img.dtype)))
        renormalized_pattern=median(img, disk(4*iloop))#ski.filters.gaussian(img, sigma=(4*iloop, 4*iloop), truncate=3.5, channel_axis=-1)
        #print(str(type(renormalized_pattern.dtype)))
	#print("success")	
        '''
        for jloop in range(2**(full_depth-iloop)):   # The core averaging procedure
            for kloop in range(2**(full_depth-iloop)):

                pixel = 1/4. * (parental_pattern[kloop*2][jloop*2]+parental_pattern[kloop*2+1][jloop*2]+parental_pattern[kloop*2][jloop*2+1]+parental_pattern[kloop*2+1][jloop*2+1] )
                renormalized_pattern[kloop][jloop] = pixel
        '''
        rp=renormalized_pattern.astype(float)
        rp=rp/np.max(rp)
        stack_of_patterns.append(rp) # Append the renormalized picture to the stack
        #print(str(img))
        #print(str(renormalized_pattern))
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
    stack = coarse_grain(img, depth1 - 3)  # Does the coarse-graining to depth = (maximal depth - 3)  -  I assume that the last three patterns are too coarse to bear any interesting information
    #print("finished cg")
    #blown_up_stack=stack
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

name = 'T_051.jpg'
name = "d_orig.png"
# Read the file

img = mpimg.imread(name)  
rs = transform.resize(img, (512,512))   
gray = rgb2gray(rs)
gray=gray/np.max(gray)
x = compute_complexities(gray)   # Compute all the complexities for the image
norm=np.einsum('ij,ij', gray, gray)
print(norm)
print("total_complexity: " + str(x))
x_n0 =x[0]/np.sqrt(norm)
x_n=x[1]/np.sqrt(norm)
gray_norm=gray/np.sqrt(norm)
x = compute_complexities(gray_norm)   # Compute all the complexities for the image
print("total_complexity normalized: " + str(x_n0))
print("partial_complexities normalized: " + str(x_n))

stack = coarse_grain(gray, 6)
#print(len(stack))
i=0
for el in stack:
	#print(np.shape(el))
	plt.clf()
	plt.imshow(el, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(el))
	plt.savefig('debug_'+str(i)+'.png')
	i=i+1

'''
	gray = rgb2gray(im)
	rs = transform.resize(gray, (512,512))
	rs=rs/np.max(rs)
	ttt = img_as_ubyte(rs)
	norm=np.einsum('ij,ij', ttt, ttt)
	#intensity of image/square root of norm
	ttt=ttt/np.sqrt(ttt)
	print("success")

'''




cf_all=[]
cs_all=[]
random_all=[]
type0_all=[]

dataset_path="/vol/tcm36/akravchenko/image_complexity/Savoias-Dataset/Images/Art/"
ranking_path="/vol/tcm36/akravchenko/image_complexity/Savoias-Dataset/Images/global_ranking/global_ranking_art.xlsx"


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
		rs=rs/np.max(rs)
		x, partial = compute_complexities(rs)

		#print(x)

		j=j+1
		complexity_channels.append(x)
		complexity_channels_partial.append(partial)

	cmpl=im_intensity[0]*complexity_channels[0] + im_intensity[1]*complexity_channels[1] + im_intensity[2]*complexity_channels[2]
	cmpl_partial=im_intensity[0]*np.asarray(complexity_channels_partial[0]) + im_intensity[1]*np.asarray(complexity_channels_partial[1]) + np.asarray(complexity_channels_partial[2])*complexity_channels[2]
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
		#break


#complexity_list=complexity_list/np.max(complexity_list)
df['gt']=df['gt'].values
df['ms_total']=complexity_list

features=[2,4,16,64,256]
df[features]=complexity_list_partial

#df = df.drop(df[df.ms_total>1000].index)

#df1=df.dropna()
'''

df['frac']=df['ms_total'].values-df['2'].values-df['64'].values

with open('sup_rank_complexity.pickle', 'wb') as handle:
    pickle.dump(df, handle)
'''

df1=df.dropna()

x=range(0,df1['ms_total'].to_numpy().shape[0])
y1=df1['gt'].to_numpy()
y2=df1['ms_total'].to_numpy()

#df_head=df1.nlargest(10, ['ms_total'], keep='first')
#df_frac=df1.drop(index=df1.nlargest(10, 'ms_total').index, inplace=True)
#y2=df['frac'].to_numpy()

'''
df_frac=df1.drop(index=df1.nlargest(10, 'ms_total').index, inplace=True)
x=range(0,df_frac['ms_total'].to_numpy().shape[0])
y1=df_frac['gt'].to_numpy()
y2=df_frac['ms_total'].to_numpy()
'''

plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title('art complexity')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('art_complexity_total_rank.png')


'''
x=range(0,df['ms_total'].to_numpy().shape[0])
y1=df['ms_total'].to_numpy()
y2=df['blur_total'].to_numpy()
#y2=df['frac'].to_numpy()
plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
plt.ylim(0,0.2)
plt.xlabel('rank filter')
plt.ylabel('blur')
plt.title('suprematism paintings complexity')
plt.legend(loc='lower right')
plt.savefig('blur comparison.png')


'''
'''
y2=df['frac'].to_numpy()
plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title('suprematism paintings complexity (frac)')
plt.legend(loc='lower right')
plt.savefig('sup_complexity_total_rank_frac.png')


y1_x = ma.masked_array(y1, mask)
y2_x = ma.masked_array(y2, mask)

plt.clf()
plt.scatter(y1_x,y2_x,color='blue', linewidth=2, alpha=0.5)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title('suprematism paintings complexity, filtered')
plt.legend(loc='lower right')
plt.savefig('sup_complexity_filtered_rank.png')
'''
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





