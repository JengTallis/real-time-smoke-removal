''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
color_balance.py

perform color balance on an image

input: image
output: color-balanced image

usage:
python3 color_balance.py
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import cv2 as cv	# OpenCV3
import numpy as np
import os
from matplotlib import pyplot as plt

'''
Color Balance Algorithms
'''

def max_white(img):
	max_bright = float(2**8)	# default
	if img.dtype == np.uint16:
		max_bright = float(2**16)
	elif img.dtype == np.uint32:
		max_bright = float(2**32)	
	img = img.transpose(2, 0, 1)	# h, w, c -> c, h, w
	img = img.astype(np.int32)		# int rep
	img[0] = np.minimum(img[0] * (max_bright/float(img[0].max())), 255)
	img[1] = np.minimum(img[1] * (max_bright/float(img[1].max())), 255)
	img[2] = np.minimum(img[2] * (max_bright/float(img[2].max())), 255)
	return img.transpose(1, 2, 0).astype(np.uint8)	#  c, h, w -> h, w, c

def gray_world(img):
	img = img.transpose(2, 0, 1).astype(np.uint32)
	mu_g = np.average(img[1])
	img[0] = np.minimum(img[0]*(mu_g/np.average(img[0])),255)
	img[2] = np.minimum(img[2]*(mu_g/np.average(img[2])),255)
	return  img.transpose(1, 2, 0).astype(np.uint8)

def gimp(img, perc = 0.05):
	for channel in range(img.shape[2]):
		mi, ma = (np.percentile(img[:,:,channel], perc), np.percentile(img[:,:,channel],100.0-perc))
		img[:,:,channel] = np.uint8(np.clip((img[:,:,channel]-mi)*255.0/(ma-mi), 0, 255))
	return img
 
def retinex(img):
	img = img.transpose(2, 0, 1).astype(np.uint32)
	mu_g = img[1].max()
	img[0] = np.minimum(img[0]*(mu_g/float(img[0].max())), 255)
	img[2] = np.minimum(img[2]*(mu_g/float(img[2].max())), 255)
	return img.transpose(1, 2, 0).astype(np.uint8)

def retinex_adjust(img):
	"""
	from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
	"""
	img = img.transpose(2, 0, 1).astype(np.uint32)
	sum_r = np.sum(img[0])
	sum_r_sq = np.sum(img[0]**2)
	max_r = img[0].max()
	max_r_sq = max_r**2
	sum_g = np.sum(img[1])
	max_g = img[1].max()
	coeff = np.linalg.solve(np.array([[sum_r_sq,sum_r],[max_r_sq,max_r]]), np.array([sum_g,max_g]))
	img[0] = np.minimum((img[0]**2)*coeff[0] + img[0]*coeff[1], 255)
	sum_b = np.sum(img[1])
	sum_b_sq = np.sum(img[1]**2)
	max_b = img[1].max()
	max_b_sq = max_r**2
	coeff = np.linalg.solve(np.array([[sum_b_sq,sum_b],[max_b_sq,max_b]]), np.array([sum_g,max_g]))
	img[1] = np.minimum((img[1]**2)*coeff[0] + img[1]*coeff[1], 255)
	return img.transpose(1, 2, 0).astype(np.uint8)

def retinex_with_adjust(img):
	return retinex_adjust(retinex(img))


'''
Test

filename = "4.jpg"
img = os.path.basename(filename)
img = cv.imread(filename)
bal = retinex_adjust(img)

#cv.namedWindow('ori', cv.WINDOW_NORMAL)
#cv.resizeWindow('ori', 100, 100)
#cv.imshow('ori',img)

#cv.namedWindow('balanced', cv.WINDOW_NORMAL)
#cv.resizeWindow('balanced', 300, 300)
#cv.imshow('balanced',bal)

# BGR (OppenCV) -> RGB (pyplot)
RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
RGB_bal = cv.cvtColor(bal, cv.COLOR_BGR2RGB)

plt.subplot(2, 1, 1)
plt.imshow(RGB_img)
plt.xticks([]), plt.yticks([])  # hide tick values on axis
# plt.axis("off")
plt.subplot(2, 1, 2)
plt.imshow(RGB_bal)
plt.xticks([]), plt.yticks([])  # hide tick values on axis
plt.show()

#cv.waitKey(0)
#cv.destroyAllWindows()
'''
