import cv2 
import pyrealsense2 as rs
import numpy as np
from realsense_depth import *

dc = DepthCamera()

while True:
    ret, depth_frame, color_frame = dc.get_frame()

    cv2.imshow("image_color", color_frame)
    cv2.imshow("image_depth", depth_frame)

    if cv2.waitKey(1) == 27:
        break

