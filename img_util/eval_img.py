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

import numpy as np
import math
from PIL import Image, ImageDraw
from ..util.gt_maker import gt_maker

'''
# =========== evaluate input image ============
'''