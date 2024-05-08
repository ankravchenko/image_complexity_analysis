import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import time
import cv2

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




# Function converting colored picture to the gray scale

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Function doing coarse graining of the image

def coarse_grain(img, depth):
    #if depth>11:#FIXME
        #depth=11
    stack_of_patterns = []  # To keep all the coarse-grained pictures
    stack_of_patterns.append(img)  # Append the original picture
    full_depth = int(np.log2(len(img))) # Estimates the total number of coarse-graining steps that can be done, assuming that we apply 2*2 filter at each step
    # do dft saving as complex output
    dft = np.fft.fft2(img, axes=(0,1))
    # apply shift of origin to center of image
    dft_shift = np.fft.fftshift(dft)
    mag = np.abs(dft_shift)
    if mag.all() != 0:
        spec = np.log(mag) / 20
    else:
        spec=0

    blur_range=np.geomspace(256.0, 1.0,num=50)#[128,96,64,48,32,24,16,12,8,4,2]
    for iloop in range(1, depth):
        # create circle mask
        r=blur_range[iloop]
        radius = int(r)
        mask = np.zeros_like(img)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        #print('preparing mask')
        cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]

        fz=radius*2-1
        # blur the mask
        #print('preparing blurry mask')
        mask2 = cv2.GaussianBlur(mask, (fz,fz), 0)
        #print('done')
        # apply mask to dft_shift
        dft_shift_masked = np.multiply(dft_shift,mask) / 255
        dft_shift_masked2 = np.multiply(dft_shift,mask2) / 255

        # shift origin from center to upper left corner
        back_ishift = np.fft.ifftshift(dft_shift)
        back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
        back_ishift_masked2 = np.fft.ifftshift(dft_shift_masked2)


        # do idft saving as complex output
        img_back = np.fft.ifft2(back_ishift, axes=(0,1))
        img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
        img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0,1))

        # combine complex real and imaginary components to form (the magnitude for) the original image again
        img_back = np.abs(img_back).clip(0,255).astype(np.float64)
        img_filtered = np.abs(img_filtered).clip(0,255).astype(np.float64)
        img_filtered2 = np.abs(img_filtered2).clip(0,255).astype(np.float64)

        renormalized_pattern=img_filtered2
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
    '''          
    depth1 = np.log2(len(img))  # Computes the possible depth of coarse graining using one side of the image
    depth2 = np.log2(len(img[0]))  # Computes the possible depth of coarse graining using the other side of the image
    
    #assert abs( int(depth1) - depth1 ) < 1e-6, "The first side should be a power of 2"
    #assert abs( int(depth2) - depth2 ) < 1e-6, "The second side should be a power of 2"

    depth1 = int(depth1)
    depth2 = int(depth2)
    '''
    #assert int(depth2) == int(depth1), "Sides must be equal"
    #print("depth="+str(depth1))
    depth1=49
    stack = coarse_grain(img, depth1)  # Does the coarse-graining to depth = (maximal depth - 3)  -  I assume that the last three patterns are too coarse to bear any interesting information
    #blown_up_stack=stack
    '''
    blown_up_stack = []      # Will store all the coarse-grained images upscaled to the original resolution
    blown_up_stack.append(img)   # Append the original image
    '''
    complexity = 0    # Will be the overall complexity
    partial = []

    #norm=np.einsum('ij,ij', img, img)
    #stack=stack/norm
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
	rs = transform.resize(gray, (512,512), anti_aliasing=False) 
	rs=rs.astype(np.float64)
	
	plt.clf()
	plt.imshow(rs, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(rs))
	plt.savefig('0debug_rs.png')
	print("resized_printed")
	###
	norm=np.einsum('ij,ij', rs, rs)
	#intensity of image/square root of norm
	rs=rs/np.sqrt(norm)

	print('start coarse grain')
	stack = coarse_grain(rs, 11)	
	print('coarse grain finished')
	print(len(stack))
	i=0
	for el in stack:
	     #print(np.shape(el))
	     plt.clf()
	     plt.imshow(el, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(el))
	     plt.savefig('debug_'+str(i)+'.png')
	     i=i+1



def regression(x,y):
	slope, intercept, r, p, std_err = stats.linregress(x, y)
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
def image_stats(im_n, im, partial_complexity, cmpl, subset_name, cg_type):
	plt.clf()
	x=range(len(partial_complexity))

	gray = rgb2gray(im)
	rs = transform.resize(gray, (512,512), anti_aliasing=False) 
	rs=rs.astype(np.float64)
	norm=np.einsum('ij,ij', rs, rs)
	#intensity of image/square root of norm
	rs=rs/np.sqrt(norm)
	stack = coarse_grain(rs, 11)

	fig, axs = plt.subplots(6, 2)
	#print("???")
	axs[0, 0].imshow(im, cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(im))
	axs[0, 0].set_title('original image, n='+str(im_n)+', total complexity='+f'{cmpl:.3f}')
	#print("!!!!")
	axs[0, 1].plot(x,partial_complexity, label='partial complexities')
	axs[0, 1].set_ylim([0, 0.03])
	axs[0, 1].set_title('FT')
	#print("you are here")

	axs[1, 0].imshow(stack[1], cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(rs))
	axs[1, 1].imshow(stack[2], cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(rs))
	axs[2, 0].imshow(stack[3], cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(rs))
	axs[2, 1].imshow(stack[4], cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(rs))
	axs[3, 0].imshow(stack[5], cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(rs))
	axs[3, 1].imshow(stack[6], cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(rs))
	axs[4, 0].imshow(stack[7], cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(rs))
	axs[4, 1].imshow(stack[8], cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(rs))
	axs[5, 0].imshow(stack[9], cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(rs))
	axs[5, 1].imshow(stack[10], cmap=plt.get_cmap('gray'), vmin=0, vmax=np.amax(rs))
	if not os.path.exists('image_debug/'+subset_name):
		os.mkdir('image_debug/'+subset_name)
	plt.savefig('image_debug/'+subset_name+'/'+str(im_n)+"_"+cg_type+"_debug.png")
	plt.close(fig)

subset_name='suprematism'
dataset_path="/vol/tcm36/akravchenko/image_complexity/Savoias-Dataset/Images/Suprematism/"
ranking_path="/vol/tcm36/akravchenko/image_complexity/Savoias-Dataset/Images/global_ranking/global_ranking_sup.xlsx"

'''
python3 complexity_colour_fft.py advertisement "../Savoias-Dataset/Images/Advertisement/" "../Savoias-Dataset/Images/global_ranking/global_ranking_ad.xlsx" fft

'''

#complexity_colour_blur.py suprematism "../Savoias-Dataset/Images/Suprematism/" "../Savoias-Dataset/Images/global_ranking/global_ranking_sup.xlsx" blur
subset_name=sys.argv[1]
dataset_path=sys.argv[2]
ranking_path=sys.argv[3]
#cg_type=sys.argv[4]
cg_type='fft'
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
		rs=rs.astype(np.float64)
		norm=np.einsum('ij,ij', rs, rs)
		#intensity of image/square root of norm
		rs=rs/np.sqrt(norm)
		x, partial = compute_complexities(rs)   # Compute all the complexities for the image
		#print(x)
		#x=x/norm
		partial=partial/norm
		complexity_list.append(x)
		complexity_list_partial.append(partial)
		local_max_list.append(1+local_max_n(partial))
		local_min_list.append(1+local_min_n(partial))
		image_stats(cnt, rs, partial, x,subset_name, cg_type)
	else:
		rs = transform.resize(im, (512,512), anti_aliasing=False) 
		im_channels=[im[:,:,0], im[:,:,1], im[:,:,2]]#FIXME there's a more elegant way to do this
		im_intensity=[0, 0, 0]
		j=0
		complexity_channels=[]
		complexity_channels_partial=[]
		for im_c in im_channels:
			norm=im_c/256
			coeff=np.sum(norm)/(norm.shape[0]*norm.shape[1]) 
			im_intensity[j]=coeff

			rs = im_c#transform.resize(im_c, (512,512), anti_aliasing=False) 
			rs=im_c.astype(np.float64)
			norm=np.einsum('ij,ij', rs, rs)
			rs=rs/np.sqrt(norm)
			#intensity of image/square root of norm
			
			x, partial = compute_complexities(rs)   # Compute all the complexities for the image
			#x=x/norm
			#partial=partial/norm
			j=j+1
			complexity_channels.append(x)
			complexity_channels_partial.append(partial)

		cmpl=(im_intensity[0]*complexity_channels[0] + im_intensity[1]*complexity_channels[1] + im_intensity[2]*complexity_channels[2])/3
		cmpl=cmpl
		cmpl_partial=(im_intensity[0]*np.asarray(complexity_channels_partial[0]) + im_intensity[1]*np.asarray(complexity_channels_partial[1]) + im_intensity[2]*np.asarray(complexity_channels_partial[2]))/3
		cmpl_partial=cmpl_partial
		complexity_list.append(cmpl)
		complexity_list_partial.append(cmpl_partial)
		local_max_list.append(1+local_max_n(cmpl_partial))
		local_min_list.append(1+local_min_n(cmpl_partial))
		image_stats(cnt, im, cmpl_partial, cmpl, subset_name, cg_type)
	cnt=cnt+1	
	#print(cnt)
	if (cnt % 10) == 0:
		print(cnt)
		#break


#complexity_list=complexity_list/np.max(complexity_list)

df['ms_total']=complexity_list
df['ms_total']=df['ms_total']*3


with open('calculated_mssc_100/'+cg_type+'_'+subset_name+'_complexity.pickle', 'wb') as handle:
    pickle.dump(df, handle)

df.to_csv('calculated_mssc_100/'+cg_type+'_'+subset_name+'_complexity.csv', sep='\t')


a=complexity_list_partial
feature_n=len(max(a,key = lambda x: len(x)))
np_partial = np.zeros([len(a),feature_n])
for i,j in enumerate(a):
    np_partial[i][0:len(j)] = j

np_partial=np_partial*3

features = [f"s{i}" for i in range(feature_n)]


df[features]=np_partial
msk=df['ms_total']>0.5 
idx = df.index[msk]
df=df.drop(idx)

avg_c=df[features].mean().to_numpy()
plt.clf()
plt.plot(range(len(avg_c)),avg_c, label='partial complexities averaged, '+subset_name)
plt.savefig('mssc_figures_100/'+cg_type+'_'+subset_name+'_partial_cmpl_avg.png')
plt.savefig('mssc_figures_eps_100/'+cg_type+'_'+subset_name+'_partial_cmpl_avg.eps', format='eps')


with open('calculated_mssc_100/'+cg_type+'_'+subset_name+'_complexity.pickle', 'wb') as handle:
    pickle.dump(df, handle)

df.to_csv('calculated_mssc_100/'+cg_type+'_'+subset_name+'_complexity.csv', sep='\t')



#removing outliers

idx=[]
if subset_name=='suprematism':
	idx=[50, 95, 64, 65, 52, 53, 13, 48, 20]
elif subset_name=='advertisement':
	msk=df['ms_total']>0.3 
	idx = df.index[msk]
elif subset_name=='art':
	idx=[380, 225, 30, 88, 119, 286, 107]
'''elif subset_name=='infographics':
	msk=df['ms_total']>0.5 
	idx = df.index[msk]'''


outliers=df.loc[idx]
df=df.drop(idx)

x=range(0,df['ms_total'].to_numpy().shape[0])
y1=df['gt'].to_numpy()
y2=df['ms_total'].to_numpy()

y1o=outliers['gt'].to_numpy()
y2o=outliers['ms_total'].to_numpy()


#human vs calculated complexity
plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
plt.scatter(y1o,y2o,color='red', linewidth=2, alpha=0.5)
#plt.ylim(0,0.2)
plt.xlabel('human ranking',fontsize=16)
plt.ylabel('multi-scale structural complexity',fontsize=16)
plt.title(subset_name+', total complexity',fontsize=20)
plt.legend(loc='lower right')
plt.savefig('mssc_figures_100/'+cg_type+'_'+subset_name+'_complexity_total.png')
plt.savefig('mssc_figures_eps_100/'+cg_type+'_'+subset_name+'_complexity_total.eps', format='eps')



#regression
x=df['ms_total']
y=df['gt']
xo=outliers['ms_total']
yo=outliers['gt']
slope, intercept, r, p, std_err = stats.linregress(x, y)
y1=slope * x + intercept
plt.clf()
plt.scatter(x, y)
plt.xlabel('human ranking',fontsize=16)
plt.ylabel('multi-scale structural complexity',fontsize=16)
plt.scatter(xo,yo,color='red', linewidth=2, alpha=0.5)
plt.plot(x, y1, color='orange')
plt.title(subset_name+', linear regression', fontsize=20)
plt.title(cg_type+' cg, '+subset_name+' regression (full set).png')
plt.savefig('mssc_figures_100/'+cg_type+'_'+subset_name+'_regression_total.png')
plt.savefig('mssc_figures_eps_100/'+cg_type+'_'+subset_name+'_regression_total.eps', format='eps')


f = open("mssc_figures_100/"+cg_type+'_'+subset_name+'_regression_total.log', "w")
ttt=[slope, intercept, r, p, std_err]
print("slope\tintercept\tr\tp\tstd_err", end='\n', file=f)
print(*ttt, sep='\t', end='\n', file=f)
f.close()

#remove outliers
#fractions
#perhaps average

'''

handle=open('calculated_mssc/'+cg_type+'_'+subset_name+'_complexity.pickle','rb')
df=pickle.load(handle)
df.to_csv('calculated_mssc/'+cg_type+'_'+subset_name+'_complexity.csv', sep='\t')


for suprematism:
idx=[50, 95, 64, 65, 52, 53, 13, 24, 23, 48, 20]
#23?
for art:
idx=[380, 225, 30, 88, 119, 286, 107]

msk=df['ms_total']>0.19 #infographics
msk=df['ms_total']>0.1 #advertisement
idx_to_drop = df.index[msk]
df=df.drop(idx_to_drop)



x=range(0,df['ms_total'].to_numpy().shape[0])
y1=df['gt'].to_numpy()
y2=df['ms_total'].to_numpy()

#human vs calculated complexity
plt.clf()
plt.scatter(y1,y2,color='blue', linewidth=2, alpha=0.5)
#plt.ylim(0,0.2)
plt.xlabel('human ranking')
plt.ylabel('multi-scale complexity')
plt.title(cg_type+' cg, '+subset_name+' total complexity (no outliers)')
plt.legend(loc='lower right')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_complexity_no_outliers.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_complexity_no_outliers.eps', format='eps')



#regression
x=df['ms_total']
y=df['gt']
slope, intercept, r, p, std_err = stats.linregress(x, y)
y1=slope * x + intercept
plt.clf()
plt.scatter(x, y)
plt.plot(x, y1, color='orange')
plt.title(cg_type+' cg, '+subset_name+' regression (no outliers).png')
plt.savefig('mssc_figures/'+cg_type+'_'+subset_name+'_regression_no_outliers.png')
plt.savefig('mssc_figures_eps/'+cg_type+'_'+subset_name+'_regression_no_otliers.eps', format='eps')


f = open("mssc_figures/"+cg_type+'_'+subset_name+'_regression_no_outliers.log', "w")
ttt=[slope, intercept, r, p, std_err]
print("slope\tintercept\tr\tp\tstd_err", end='\n', file=f)
print(*ttt, sep='\t', end='\n', file=f)
f.close()
'''



