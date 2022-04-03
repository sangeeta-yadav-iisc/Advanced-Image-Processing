import numpy as np
import os
from scipy.fftpack import dctn,idctn
import skimage.io as io
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import time
import sys
from tqdm import tqdm
from skimage.metrics import mean_squared_error as mse


class jpeg:
	def __init__(self,image = None):
		self.I = image
		
		self.Q = np.array([[20,10,40,40,40,40,40,40],
						   [10,10,40,40,40,40,40,40],
						   [40,40,40,40,40,40,40,40],
						   [40,40,40,40,40,40,40,40],
						   [40,40,40,40,40,40,40,40],
						   [40,40,40,40,40,40,40,40],
						   [40,40,40,40,40,40,40,40],
						   [40,40,40,40,40,40,40,40]])
		
		self.X_HAT = np.zeros(self.I.shape)
		self.X_HAT_DCT = np.zeros(self.I.shape)
		self.compressed = None
		self.compressed_size = 0
		self.SIZE = os.path.getsize('cameraman.tif')
	
	@classmethod
	def DCT(cls,patch):
		return dctn(patch, type = 2)

	@classmethod
	def split( cls, img, window_size = [8,8], step_size = 8):
	    image = img
	    for y in range(0,image.shape[0]-window_size[1]+1,step_size):
	        for x in range(0,image.shape[1]-window_size[0]+1,step_size):
	            yield y,x,image[y:y+window_size[1],x:x+window_size[0]]

	def jpeg(self, quantize = True):
		reconstructed_dct_img = []
		
		if(quantize is True):
			for y,x,patch in jpeg.split(self.I):
				I_dct = jpeg.DCT(patch)
				Quantized_I_dct = np.floor((I_dct / self.Q) + 0.5).astype(np.int16)
				reconstructed_dct_img.append((y,x,Quantized_I_dct))
		else:
			for y,x,patch in jpeg.split(self.I):
				I_dct = jpeg.DCT(patch)
				Quantized_I_dct = np.round_(I_dct).astype(np.int16)
				reconstructed_dct_img.append((y,x,Quantized_I_dct))
		
		X_HAT = np.zeros(self.I.shape)
		
		for ri in reconstructed_dct_img:
			r = ri[0]
			c = ri[1]
			x_hat_patch = ri[2]
			self.X_HAT_DCT[r:r+8,c:c+8] = x_hat_patch
			X_HAT[r:r+8,c:c+8] = idctn(x_hat_patch*self.Q)

		self.X_HAT = X_HAT

		return self.X_HAT

	def show(self,*args):
		plt.imshow(self.X_HAT, cmap = 'gray')
		if(args):
			plt.title(args[0])
		plt.show()

	
	def encode_single(self,x):
		if(x == 0):
			return '0'
		x = np.int16(x)
		l = int(np.log2(np.abs(x)))
		if(x < 0):
			b = np.binary_repr(x, width = l+2)
		else:
			b = np.binary_repr(x, width = l+1)
		#print((l+1)*'1' + '0' + b)
		return (l+1)*'1' + '0' + b

	def encode(self, verbose = True):
		'''
		encoding rule:
					0 				---> 0
				 -1   1 			---> 10x
			  -3 -2   2  3 			---> 110xx
		  -7 -6 -5 -4  4  5  6  7 	---> 1110xxx
		          ... 				---> ...

		  indices in (-2^(i-1) , -2^i +1) ∪ (2^(i-1) , 2^i -1) is represented using i 1's and one 0 followed by i encoding symbols
		'''
		self.compressed = np.empty(self.X_HAT_DCT.shape, dtype = 'object')
		
		for r in range(self.I.shape[0]):
			for c in range(self.I.shape[1]):
				self.compressed[r,c] = self.encode_single(self.X_HAT_DCT[r,c])

		bitstream = ""

		for e in self.compressed.ravel():
			bitstream+= e

		

		compressed = bytearray()
		for i,_ in enumerate(bitstream):
			compressed.append(int(bitstream[i]))

		
		self.compressed_size = len(bitstream)/8
		
		if(verbose is True):
			details = f"actual image size  {os.path.getsize('cameraman.tif') :>35} bytes" + f"\nSize of bitstream(file) after compression  {len(bitstream)/8 :>12} bytes"\
																						+ f"\ncompression ratio {round(8*os.path.getsize('cameraman.tif')/len(bitstream),2)  :>35} "
			self.compressed = compressed
		
			return self.compressed,self.compressed_size, details
		else:
			return self.compressed,self.compressed_size

	def getQ(self,a,b,c):
		Q = np.array([[c,a,b,b,b,b,b,b],
				      [a,a,b,b,b,b,b,b],
				      [b,b,b,b,b,b,b,b],
				      [b,b,b,b,b,b,b,b],
				      [b,b,b,b,b,b,b,b],
				      [b,b,b,b,b,b,b,b],
				      [b,b,b,b,b,b,b,b],
				      [b,b,b,b,b,b,b,b]])
		return Q

	def calc_Q(self, refsize):
		print('\n Please Wait... Calculating Optimal Q')
		l = np.arange(10,110,10)

		ii,jj,kk = np.meshgrid(l,l,l)

		choice = dict()

		for t,i,j,k in zip(tqdm(range(len(l)**3)),ii.ravel(),jj.ravel(),kk.ravel()):
			self.Q = self.getQ(i,j,k)
			self.jpeg()
			_,size = self.encode(verbose = False)
			if(size < refsize):
				#print(size)
				choice.update({mse(self.X_HAT,self.I): [np.array([i,j,k]), size]})

		min_mse = sorted(choice)[0]
		#keys = choice.keys()
		
		print(f'minimum mse : {min_mse} with (a,b,c) = {choice[min_mse][0]} and compressed size = {choice[min_mse][1]} bytes')
		return 

if __name__ == '__main__':
	image = io.imread('cameraman.tif').astype('uint8')
	if(len(image.shape) == 3):
		image = rgb2gray(image)

	j = jpeg(image)

	start = time.time_ns()
	JPEG_image = j.jpeg()
	stop = time.time_ns()
	print(f'JPEG conversion time  {(stop - start)//1e6} milliseconds')

	print('\n\t\tWith Quantization')
	print('___________________________________________________________________')

	
	#j.show('Image after Quantization')
	
	start = time.time_ns()
	_, size,details = j.encode()
	stop = time.time_ns()
	print(details)
	print(f'encoding time  {(stop - start)//1e6 :>35} milliseconds')
	print(f'mean squared error  {mse(JPEG_image,image)/255**2 :>42}')
	print('___________________________________________________________________')

	print('\n')
	
	J = j.jpeg(quantize = False)
	#j.show('Image with rounded DCT coefficients')

	start = time.time_ns()
	_,_i,details = j.encode()
	stop = time.time_ns()
	print('\n\t\tWithout Quantization(Rounding off)')
	print('___________________________________________________________________')
	print(details)
	print(f'encoding time : {(stop - start)//1e6 :>33} milliseconds')
	print(f'mean squared error : {mse(J,image) :>40}')
	print('___________________________________________________________________')
	
	j.calc_Q(refsize = size)