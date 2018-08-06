''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cap_frame.py

extract frames from the input videos 

input: video
output: frames of the video

usage:
python3 get_frame.py
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import time
import os
import cv2 as cv	# OpenCV3

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

video2frames(VIDEO, OUT_DIR)
		