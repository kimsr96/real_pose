from matplotlib import pyplot as plt
import numpy as np
import pyrealsense2 as rs
import cv2 
import numpy as np
import mediapipe as mp
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

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    aligned_color_frame = aligned_frames.get_color_frame()

    #fill the Holes
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.holes_fill, 3)
    filtered_depth = spatial.process(depth_frame)
    hole_filling = rs.hole_filling_filter()
    filled_depth = hole_filling.process(filtered_depth)

    depth_image = np.asanyarray(filled_depth.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    image_height, image_width, _ = color_image.shape

    results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

    if not depth_frame or not color_frame: 
        continue
    
    arr_x, arr_y, arr_z = [], [], []

    if results.pose_landmarks:
        for i in range(11, 33):
            if results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility >= 0.995:
                x = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width
                y = results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height
                if y >= 480:
                    continue
                depth = depth_frame.get_distance(int(x), int(y))
                print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}') 
                dx ,dy, dz = rs.rs2_deproject_pixel_to_point(color_intrin, [x,y], depth)
                distance = math.sqrt(((dx)**2) + ((dy)**2) + ((dz)**2))
                if distance == 0:
                    continue
                print(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * image_width)
                arr_x.append(x)
                arr_y.append(y)
                arr_z.append(distance * 100)
            
            else:
                continue
            

        #print(arr_x, arr_y, arr_z)
        # arr_x = [x for x in arr_x if x!= 0]
        # arr_y = [y for y in arr_y if y!= 0]
        # arr_z = [z for z in arr_z if z!= 0]
        
        print(arr_x)
        print(arr_y)
        print(arr_z)
        print("=================")
        plt.rcParams['figure.figsize'] = (8, 6)
        ax = plt.axes(projection = '3d')
        ax.scatter3D(arr_x, arr_z, arr_y)

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        
        mp_drawing.draw_landmarks(image=color_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", color_image)
        plt.show()
    if cv2.waitKey(1) == 27:
        break