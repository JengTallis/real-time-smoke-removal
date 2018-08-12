''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cap_frame.py

extract frames from the input videos 

input: video
output: frames of the video

usage:
python3 get_frame.py

Usage: 
Under 	parent of real-time-smoke-removal
Run 	python3 -m real-time-smoke-removal.vid_util.get_frame
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import time
import os
import cv2 as cv	# OpenCV3
import math
from ..img_util.smoke_restore import smoke_restore


''' 
Set relative path to absolute
'''
here = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), x))

video = '../ori_vid/shen-1.mp4'
out_dir = '../frames/'

VIDEO = here(video)
OUT_DIR = here(out_dir)


'''
# =========== extract frmes from video and put ============
'''
def video2frames(video, out_dir):
	t_start = time.time()			# log start time
	cap = cv.VideoCapture(video)	# capture the feed
	try:
		os.mkdir(out_dir)			# make output directory
	except OSError:
		pass
	vid_len = int (cap.get(cv.CAP_PROP_FRAME_COUNT)) - 1		# number of frames in the video
	print ("Num of frames: ", vid_len)
	count = 0						# count the number of frames processed
	while cap.isOpened():
		ret, frame = cap.read()		# extract frame
		cv.imwrite(out_dir + "/%#05d.jpg" % (count+1), frame)	# write frame to img output
		count += 1
		if (count > (vid_len-1)):
			time_end = time.time()	# log end time
			cap.release()			# release the feed
			print ("Frame extraction finished.\n %d frames extracted" % count)
			print ("Extraction time: %d seconds." % (time_end-time_start))
			break

def real_time_desmoke_video(video):
	cap = cv.VideoCapture(video)	# capture the feed
	ret, frame = cap.read()
	(height, width) = frame.shape[:2]
	scale = 0.35
	h = math.floor(height*scale)
	w = math.floor(width*scale)
	while(cap.isOpened()):
		rs_frame = cv.resize(frame, (w, h)) 
		res = smoke_restore(rs_frame, 0.75, 7, 3, 0.95, 1.3)
		# --- display desmoked img ---
		cv.namedWindow('desmoke', cv.WINDOW_NORMAL)
		cv.resizeWindow('desmoke', (w, h))
		cv.imshow('desmoke', res)
		# --- display original img ---
		cv.namedWindow('ori', cv.WINDOW_NORMAL)
		cv.resizeWindow('ori', (w, h))
		cv.imshow('ori', rs_frame)

		ret, frame = cap.read()
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv.destroyAllWindows()

real_time_desmoke_video(VIDEO)
#video2frames(VIDEO, OUT_DIR)
		