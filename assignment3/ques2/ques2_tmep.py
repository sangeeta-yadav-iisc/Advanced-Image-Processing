import cv2, os
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('lighthouse2.bmp')
cv2.imwrite('image.png', img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('grayimage.png', gray_img)
noise = gray_img.copy()
cv2.randn(noise,(0),(10)) 
# temp_noise = noisy_img.copy()
noisy_img = gray_img + noise
noisy_img[noisy_img > 255] = 255
noisy_img[noisy_img < 0] = 0
cv2.imwrite('noisyimage.png', noisy_img)
# cv2.imshow("GrayScaledOriginal",gray_img)
# cv2.waitKey(3000)
# cv2.imshow("NoisyImage",noisy_img)
# cv2.waitKey(3000)

def gaussian_filter(shape =(3,3), sigma=1):
    x, y = [edge //2 for edge in shape]
    grid = np.array([[((i**2+j**2)/(2.0*sigma**2)) for i in range(-x, x+1)] for j in range(-y, y+1)])
    g_filter = np.exp(-grid)/(2*np.pi*sigma**2)
    g_filter /= np.sum(g_filter)
    return g_filter

## Part 1 Low pass Gaussian Filter
filterlens = [3,7,11]
sigmas = [0.1,1,2,4,8]
print(f"Filter \t Sigma \t MSE")
MSEs = []
filterind = []
for filterlen in filterlens:
    for sigma in sigmas:
        filters = gaussian_filter(shape =(filterlen,filterlen), sigma=sigma)
        flt_img = cv2.filter2D(src=noisy_img, ddepth=-1, kernel=filters)
        Y = np.square(np.subtract(gray_img,flt_img)).mean()
        filterind.append([filterlen,sigma])
        MSEs.append(Y)
        print(f"{filterlen},\t{sigma},\t{Y}")
min_kernel=filterind[MSEs.index(min(MSEs))]
print(f"The least MSE error is {min(MSEs):5f} for the filter size and sigma: {min_kernel}")

## Show denoised image from best low pass gaussian filter
filters = gaussian_filter(shape =(min_kernel[0],min_kernel[0]), sigma=10)#min_kernel[1])
flt_img = cv2.filter2D(src=noisy_img, ddepth=-1, kernel=filters)
# cv2.imshow('low_pass_filtered_image',flt_img)
# cv2.waitKey(5000)  
cv2.destroyAllWindows() 


## Part 2 MMSE Filter on high pass coefficients
hpf = -np.ones((3,3))
hpf[1,1] = 8

y1 = cv2.filter2D(src=noisy_img, ddepth=-1, kernel=hpf)
var_z1 = 100*((1-hpf[0,0])**2+((hpf**2).sum()-hpf[0,0]**2))

# mu_y = np.mean(noisy_img)
var_x1 = np.var(y1)-var_z1
denoised_img = flt_img+(var_x1*y1)/(var_x1+var_z1)

cv2.imshow('filtered_image_MMSE',denoised_img)
cv2.waitKey(5000)  
cv2.destroyAllWindows() 
Y = ((gray_img-denoised_img)**2).mean()
print(Y)

# plt.imshow(gray_img)
# plt.show()
# plt.imshow(denoised_img)
# plt.show()

# ### Adaptive MSME filters
# nrows, ncols = y1.shape 
 
# mask = np.zeros(y1.shape,np.uint8)
# for row in range(ncols-11+1):
#     for col in range(nrows-11+1):
#         patch = y1[row:row+11,col:col+11]
#         var_x1 = np.var(y1)-var_z1
#         denoised_img = flt_img+(var_x1*y1)/(var_x1+var_z1)

#         mask[row:row+11,col:col+11]+=patch





