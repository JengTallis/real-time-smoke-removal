''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
smoke_restore.py

perform restoration-based smoke removal on an image

input: smoky_image
output: desmoked_image

usage:
python3 smoke_restore.py
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import cv2 as cv		# OpenCV3
import numpy as np
import os
import math
from matplotlib import pyplot as plt

#import color_balance # in this
#from eval_img import brightness_gain # in this
from . import color_balance # in vid util

'''
# --- Inputs ---
img: original image
p: percentage of restoration
d: maximum size of assumed white object
balance: 0 for no white balance, 1 for white balance
max_window: maximum window size for adapted filtering
g: factor for gamma correction to achieve more colorful result

# --- Output ---
res: restored image
'''

def smoke_restore(img, p, d, max_window, balance, g):
	# === Check if input is out of bound ===
	max_window = math.floor(max_window)
	if max_window < 1:
		error("max_window out of bound")	# max_window ∈ N, max_window >= 1
	d = math.floor(d)
	if d < 1:
		error("d out of bound")		# d ∈ N, d >= 1
	if not (p <= 1 and p >= 0):
		error ("p out of bound")			# 0 <= p <= 1

	# === Differentiate Monchrome and Color Image ===
	(height, width, c) = img.shape

	# === Color Balance ===
	if c == 3:
		img = color_balance.retinex_adjust(img)	# color balance -> Is becomes (1,1,1)

	# === Normalize I in [0,1] ===			
	max_bright = float(2**8)	# default
	if img.dtype == np.uint16:
		max_bright = float(2**16)
	elif img.dtype == np.uint32:
		max_bright = float(2**32)
			
	img = (img / (max_bright - 1)).astype(np.float32)	# normalize intensity: min 0, max 1
	assert(img.max() <= 1)

	# === Compute Photometric Bound ===
	if c == 3:
		w = img.min(2)			# dimension: h, w (2D)
		mo = img.mean(2)		# dimension: h, w (2D)
	else:
		w = img
		mo = img

	# === Compute Saturation Bound ===
	sigmaColor = 250
	sigmaSpace = 150

	# --- Version 1: cv.medianBlur(src, d) ---
	#w = w.astype(np.uint8)
	#a = cv.medianBlur(w, d)		# d should be odd positive int
	#b = a - cv.medianBlur((np.absolute(w - a)), d)

	# --- version 2: cv.bilateralFilter(src, d, sigmaColor, sigmaSpace) ---
	a = cv.bilateralFilter(w, d, sigmaColor, sigmaSpace)							# a = bil(W)
	b = a - cv.bilateralFilter((np.absolute(w - a)), d, sigmaColor, sigmaSpace)		# b = a - bil(|w-a|)
	# === Infer V(x,y) with w & b bounds ===
	#print("Atmospheric veil")
	v = p * np.maximum(np.minimum(w,b), 0)											# v = max(min(pb, w),0)

	# === Restore R with Inverse Koschmieder Law ===
	#print("Restoration")
	r = np.zeros(img.shape)						# restored img
	ones = np.ones(v.shape)
	fac = np.divide(ones, (ones - v))			# f = 1/(1 - v)

	if c == 3:
		r[:,:,0] = np.multiply((img[:,:,0] - v), fac)
		r[:,:,1] = np.multiply((img[:,:,1] - v), fac)
		r[:,:,2] = np.multiply((img[:,:,2] - v), fac)
		mr = r.mean(2)
		
	else:
		r = np.multiply((img - v), fac)			# r = (I - v) * (1/(1 - v)) = (I - v) * f
		nbr = r

	# === Adapted Smoothing ===
	if max_window > 1:
		smooth_r = cv.medianBlur(r.astype(np.float32), max_window)
		r = smooth_r
		mr = r.mean(2)

	#return r

	# === Gamma Correction (bottom 1/3 of the original) ===
	lo = np.log(mo[math.floor(height*2/3):height,:]+0.5/225)	# adding a small const to avoid log(0) error
	lr = np.log(mr[math.floor(height*2/3):height,:]+0.5/225)
	mlo = np.mean(lo)
	mlr = np.mean(lr)
	slo = np.std(lo)
	slr = np.std(lr)
	pwr = g * slo / slr
	u = np.power(r, pwr) * math.exp(mlo-mlr*pwr)

	# === Tone Mapping ===
	mg = np.max(u)
	t = np.divide(u, 1+(1.0-1.0/mg)*u)

	# === return restored img ===
	return t


max_bright = float(2**8)-1
for i in range(1,13):
	filename = str(i) + '.jpg'
	img = os.path.basename(filename)
	img = cv.imread(filename)
	res = smoke_restore(img, 0.70, 17, 3, 0.95, 1.3)
	r = res * max_bright
	cv.imwrite(str(i) + 'r' + '.jpg', r)
	# write gray images
	gimg = cv.cvtColor( img, cv.COLOR_RGB2GRAY )
	cv.imwrite(str(i) + 'og' + '.jpg', gimg)
	gres = cv.cvtColor( r, cv.COLOR_RGB2GRAY )
	cv.imwrite(str(i) + 'rg' + '.jpg', gres)

'''
filename = '1.jpg'
img = os.path.basename(filename)
img = cv.imread(filename)
res = smoke_restore(img, 0.70, 17, 3, 0.95, 1.3)
r = res * max_bright
cv.imwrite('1r.jpg',r)

RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
RGB_res = cv.cvtColor(res, cv.COLOR_BGR2RGB)

gain = brightness_gain(img, res)
print("Contrast gain")
print(gain)

plt.subplot(2, 1, 1)
plt.imshow(RGB_img)
plt.xticks([]), plt.yticks([])  # hide tick values on axis
# plt.axis("off")
plt.subplot(2, 1, 2)
plt.imshow(RGB_res)
plt.xticks([]), plt.yticks([])  # hide tick values on axis
plt.show()
'''

