import pyrealsense2 as rs
import cv2 
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math

mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

align_to = rs.stream.depth
align = rs.align(align_to)

#Get rgb
rgb_list = []

while True:
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    image_height, image_width, _ = color_image.shape

    if not depth_frame: 
        continue

    results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        # Iterate two times as we only want to display first two landmarks.
        for i in range(33):
            
            pos_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width)
            pos_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height)
            
            rgb_list = []

            if pos_x >= 640 or pos_y >= 480:
                continue

            if i == 11 or i == 12 or i == 23 or i == 24:
                print(pos_x, pos_y)
                print(f'{mp_pose.PoseLandmark(i).name}:') 
                print(color_image[pos_y - 10 : pos_y + 10, pos_x - 10 : pos_x + 10])
                rgb_list.append(color_image[pos_y - 1 : pos_y + 1, pos_x - 1 : pos_x + 1])
                cv2.imshow("image_depth", color_image[pos_y - 1 : pos_y + 1, pos_x - 1 : pos_x + 1])

            print(rgb_list)
            # print(f'{mp_pose.PoseLandmark(i).name}:') 
            # print(f'x: {pos_x}')
            # print(f'y: {pos_y}')
            # print(f'z: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z}')
            #print(f'distance: {dist}')
        
        mp_drawing.draw_landmarks(image=color_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)

    #cv2.imshow("image_depth", color_image)       

    if cv2.waitKey(1) == 27:
        break