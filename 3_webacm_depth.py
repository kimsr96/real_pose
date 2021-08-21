import cv2 
import pyrealsense2 as rs
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math
from realsense_depth import *

dc = DepthCamera()
mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

while True:
    ret, depth_frame, color_frame = dc.get_frame()
    image_height, image_width, _ = color_frame.shape
    cv2.imshow("image_color", color_frame)
    cv2.imshow("image_depth", depth_frame)

    results = pose.process(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        # Iterate two times as we only want to display first two landmarks.
        for i in range(2):
            pos_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width
            pos_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height
            print(f'{mp_pose.PoseLandmark(i).name}:') 
            print(f'x: {pos_x}')
            print(f'y: {pos_y}')
            print(f'z: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * image_width}')

            # print(f'visibility: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility}\n')

    if cv2.waitKey(1) == 27:
        break



