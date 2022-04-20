from PIL import Image
import numpy as np
import os
import cv2

img = cv2.imread('lighthouse2.bmp')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray_image',gray_img)
# cv2.waitKey(3000)  

noise = gray_img.copy()
cv2.randn(noise,(0),(10)) 
noisy_img = gray_img + noise

def gaussian_filter(shape =(3,3), sigma=1):
    x, y = [edge //2 for edge in shape]
    grid = np.array([[((i**2+j**2)/(2.0*sigma**2)) for i in range(-x, x+1)] for j in range(-y, y+1)])
    g_filter = np.exp(-grid)/(2*np.pi*sigma**2)
    g_filter /= np.sum(g_filter)
    return g_filter

# filters = gaussian_filter(shape =(3,3), sigma=10)#min_kernel[1])
flt_img = cv2.GaussianBlur(noisy_img, (3,3), 1.0)
# flt_img = cv2.filter2D(src=noisy_img, ddepth=-1, kernel=filters)
cv2.imwrite('Low_pass_image.png',flt_img)
cv2.imshow('Low_pass_image',flt_img)

cv2.waitKey(3000)  
hpf = np.array([(-1,-1,-1),(-1,8,-1),(-1,-1,-1)])


def sharpen_img(src_img, gain=1.0):
    
    hpf_img = cv2.filter2D(src=flt_img, ddepth=-1, kernel=hpf)
    sharp_img = flt_img+gain*hpf_img
    mse = np.mean((src_img-sharp_img)**2)
    return mse,sharp_img

mses = []
min_mse = np.infty
for gain in range(1,10):
    mse,_ = sharpen_img(gray_img, gain=gain)
    mses.append(mse)
    if mses[-1]<min_mse:
        min_mse =mses[-1]
        best_gain = gain

_, sharp_img = sharpen_img(gray_img, gain=50)
sharp_img[sharp_img > 255] = 255
sharp_img[sharp_img < 0] = 0
cv2.imwrite('sharpimage_gain50.png', sharp_img)
cv2.imshow('Sharp_image',sharp_img)

cv2.waitKey(5000)  
cv2.destroyAllWindows() 
print(f"The min MSE = {min_mse} is achieved for gain = {best_gain}")


