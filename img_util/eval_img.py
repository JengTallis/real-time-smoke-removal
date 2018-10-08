''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
eval_img.py

evaluate the visual quality of the input image

input: image
output: an evaluation score

usage:
python3 eval_img.py [img]
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

import sys
import os
import numpy as np
import cv2 as cv	# OpenCV3
import math
from PIL import Image, ImageStat

'''
# =========== image brightness ============
'''

def brightness(img):	# scalar
	#im = Image.open(img)
	im = Image.fromarray(np.uint8(img*255))
	stat = ImageStat.Stat(im)
	r, g, b = stat.rms
	return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))	# perceived brightness

def brightness_gain(ori, processed):
	return (brightness(processed) - brightness(ori))


'''
# =========== image contrast ============
'''
def s_kernel(band_img, x, y, ch, p, m, h, w):
	s = 0
	for i in range(-p, p + 1):
		for j in range(-p, p + 1):
			if x + i < 0:
				a = 0
			else:
				a = min(h-1, x+i)
			if y+j < 0:
				b = 0
			else:
				b = min(w-1, y+j)
			s += abs(int(band_img[a, b, ch]) - int(m[x, y, ch]))
	return s


def contrast_img(band_img, p):	# input: 3D img
	(h, w, b) = band_img.shape
	ksize = 2 * p + 1
	kernel = np.ones((ksize,ksize),np.float32)/(ksize**2)
	m = cv.filter2D(img, -1, kernel)
	s = np.ones((h, w, b),np.float32)
	for ch in range(b):
		for i in range(h):
			for j in range(w):
				s[i,j,ch] = 1/(ksize**2) * s_kernel(band_img, i, j, ch, p, m, h, w)
	c = np.divide(s, m)
	return c

def contrast(img, p):	# input: 3D color img
	cimg = contrast_img(img, p)
	c = cimg.mean(2)
	return c

def mean_contrast(img, p):		# scalar
	(m, n) = img.shape[:2]
	mean_contrast = np.sum(contrast(img, p))/m/n
	return mean_contrast

def contrast_gain(ori, proc, p):
	return (mean_contrast(proc, p) - mean_contrast(ori, p))


'''
# =========== mean ratio of gradient ============
'''




'''
filename = "4.jpeg"
img = os.path.basename(filename)
img = cv.imread(filename)
bright = brightness(img)
print(bright)
p = 1
mcon = mean_contrast(img, p)
print(mcon)
'''
