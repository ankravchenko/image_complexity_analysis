import numpy as np
import cv2

from matplotlib import pylab as plt
import skimage as ski
from skimage import io, transform

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# read input and convert to grayscale
img = cv2.imread('10.jpg')
gray = rgb2gray(img)
rs = transform.resize(gray, (512,512), anti_aliasing=False) 
img=rs#img=rs/np.sqrt(norm)
norm=np.einsum('ij,ij', rs, rs)
img=img/np.sqrt(norm)
# do dft saving as complex output
dft = np.fft.fft2(img, axes=(0,1))

# apply shift of origin to center of image
dft_shift = np.fft.fftshift(dft)

# generate spectrum from magnitude image (for viewing only)
mag = np.abs(dft_shift)
spec = np.log(mag) / 20

blur_range=[256,128,64,32,16,8,4,2]

fig, axes = plt.subplots(nrows=len(blur_range), ncols=4,figsize=(40, 40))



for i in range(len(blur_range)):
	# create circle mask
	r=blur_range[i]
	g_blur=2**(i+1)
	radius = r
	mask = np.zeros_like(img)
	cy = mask.shape[0] // 2
	cx = mask.shape[1] // 2
	cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]

	fz=r*2-1
	# blur the mask
	mask2 = cv2.GaussianBlur(mask, (fz,fz), 0)

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

	img_gauss=renormalized_pattern=ski.filters.gaussian(img, sigma=(g_blur, g_blur), truncate=3.5, channel_axis=-1)
	img_gauss_tr=renormalized_pattern=ski.filters.gaussian(img, sigma=(g_blur, g_blur), truncate=i, channel_axis=-1)

	# write result to disk
	plt.sca(axes[i, 0])
	plt.axis('off')
	plt.imshow(img_filtered, cmap='gray')
	plt.sca(axes[i, 1])
	plt.axis('off')
	plt.imshow(img_filtered2, cmap='gray')
	plt.sca(axes[i, 2])
	plt.axis('off')
	plt.imshow(img_gauss, cmap='gray')
	plt.sca(axes[i, 3])
	plt.axis('off')
	plt.imshow(img_gauss_tr, cmap='gray')


	#cv2.imwrite("17_"+str(r)+"_filtered2.png", img_filtered1)
	#cv2.imwrite("17_"+str(r)+"_filtered2.png", img_filtered2)

plt.savefig('fft_vs_gauss_line.png')
