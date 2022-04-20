import skimage.io as io
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.color import rgb2gray




def LP_filt(gray_img,noisy_img,L,sd):
    # Part 1: Doing the low pass filtering with different filter length and sigma
    print("\tPart1: Low Pass Filtering...\n")
    mse = dict()
    for l in L:
        for s in sd:
			mse.update({np.mean((gauss_filt(noisy_img,sigma = s, length = l).ravel()-gray_img.ravel())**2):[l,s]})
	sorted_keys = sorted(mse)

	print(f"Min. MSE is found for σ = {mse[sorted_keys[0]][1]} and filter length = {mse[sorted_keys[0]][0]}")
	filtered_img = gauss_filt(noisy_img,sigma = mse[sorted_keys[0]][1], length = mse[sorted_keys[0]][0])
	io.imsave('lp_image.jpg',filtered_img)
	plt.imshow(filtered_img,cmap = 'gray')
	plt.title(f'Gaussian LPF with σ = {mse[sorted_keys[0]][1]}, &\nFilter length = {mse[sorted_keys[0]][0]}')
	plt.show()

def gauss_filt(I,sigma = 1, length = 3):
	return cv2.GaussianBlur(I,ksize = (length,length),sigmaX = sigma, borderType = 0)

def MMSE(L, sd,image = None):
	hp_noise = image - gauss_filt(image, sd,L)
	sd_orig = np.var(hp_noise.ravel()) - 100
	return np.mean(image.ravel()) + (sd_orig/(sd_orig+100))*image

def MMSE_filt(gray_img,noisy_img,L,sd):
    print("\tPart2: MMSE Filter...\n")
	mse = dict()
	for l in L:
		for s in sd:
			clean_img = MMSE(l,s,noisy_img)
			# original_img = get_images()[0]
			mse.update({np.mean(clean_img.ravel()-gray_img.ravel())**2 : [clean_img, s, l]})
	
	sorted_keys = sorted(mse)
	print(f"least MSE occurs for σ = {mse[sorted_keys[0]][1]} and filter length = {mse[sorted_keys[0]][2]}")
	plt.imshow(mse[sorted_keys[0]][0],cmap = 'gray')
	plt.title(f'MMSE with σ = {mse[sorted_keys[0]][1]}, & filter length = {mse[sorted_keys[0]][2]}')
	plt.show()

def get_patch( image, path_size = [11,11], step = 6):
    for y in range(0,image.shape[0]-path_size[1]+1,step):
        for x in range(0,image.shape[1]-path_size[0]+1,step):
            yield y,x,image[y:y+path_size[1],x:x+path_size[0]]

def Ada_MMSE(patch_size,overlap,img = None):
	print("\tPart3: Adaptive_MMSE...\n")
	step = patch_size-overlap
	layers = []
    for i,patch in zip(list(range(0,img.size,step)),get_patch(img, [patch_size,patch_size], step)):
		filtered_patch = MMSE(length = patch_size,σ = 10,image = patch[2])
		layer = np.zeros(img.shape)
		layer[patch[0]:patch[0]+patch_size,patch[1]:patch[1]+patch_size] = filtered_patch
		layers.append(layer)

	Filter = layers[0]
	for i,_ in enumerate(Filter):
		for j,_ in enumerate(Filter[i]):
			if(i > 10 or j > 10):
				S = 0
				nz_lay = 0 ## No. of non zero layers
				for layer in layers:
					S+= layer[i,j]
					if(layer[i,j] != 0):
						nz_lay+= 1
					if (nz_lay):
						Filter[i,j] = S/nz_lay
					else:
						Filter[i,j] = 0
	Filter = Filter.astype(np.uint8)
	return Filter





if __name__ == '__main__':

    #Read the image 

    image = (rgb2gray(io.imread('lighthouse2.bmp'))*255).astype('uint8')
	noisy_image = (np.random.normal(loc = 0, scale = 10, size = image.shape) + image).astype(np.uint8)

    # Part 1: Doing the low pass filtering with different filter length and sigma
    filter_lengths = [3, 7, 11]
	sd = [0.1, 1, 2, 4, 8]
	LP_filt(image, noisy_image,filter_lengths,sd) 
	
    # Part 2: MMSE filter 
    MMSE_filt(image,noisy_image,L,sd) 

    # Part3: Adaptive MMSE filter for path of 11 x11 and overlap of 5
	I = Ada_MMSE(11,5)
	plt.imshow(I,cmap = 'gray')
	plt.show()




