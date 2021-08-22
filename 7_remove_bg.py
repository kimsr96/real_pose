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

profile = pipeline.start(config)

align_to = rs.stream.depth
align = rs.align(align_to)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 3 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
try:
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        image_height, image_width, _ = color_image.shape

        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #images = np.hstack((bg_removed, depth_colormap))


        if not depth_frame: 
            continue

        results = pose.process(cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            #Iterate two times as we only want to display first two landmarks.
            for i in range(10):
                pos_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width)
                pos_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height)
                print(pos_x, pos_y)
                dist = depth_frame.get_distance(int(pos_x), int(pos_y))
                
                if dist == 0:
                    continue
                
                else:
                    print(f'{mp_pose.PoseLandmark(i).name}:') 
                    print(f'x: {pos_x}')
                    print(f'y: {pos_y}')
                    print(f'z: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z}')
                    print(f'distance: {dist}')
                    cv2.putText(color_image,  f'distance: {dist}', (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.circle(color_image, (pos_x, pos_y), 8, (0,0,255), -1)
                print(f'visibility: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility}\n')
        
            mp_drawing.draw_landmarks(image=bg_removed, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)

        
        # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        # cv2.imshow("Align Example", images)
        cv2.imshow("image_depth", bg_removed)

        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
