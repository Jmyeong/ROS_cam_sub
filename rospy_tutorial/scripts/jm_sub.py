#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
import imutils
import rospkg
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge, CvBridgeError  # 추가

image_data = None
distance = 0
def callback_msg(msg):
    global image_data
    try:
        # CvBridge를 사용하여 sensor_msgs/Image 메시지를 OpenCV 이미지로 변환
        bridge = CvBridge()
        image_data = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # 여기서 이미지 데이터를 가지고 원하는 처리를 수행
        # 예: 이미지 표시
        #cv2.imshow("Received Image", image_data)
        #cv2.waitKey(1)

    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: %s", e)
        
def callback_distance(distance_msg):
    global distance
    distance = distance_msg.data

if __name__ == '__main__':
    rospy.init_node('msg', anonymous=True)

    rospy.Subscriber('msg', Image, callback_msg, queue_size=10)
    rospy.Subscriber('distance_msg', Int32, callback_distance, queue_size=10)
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('webcamss')
    
    model_file = os.path.join(package_path, 'enet-cityscapes', 'enet-model.net')
    class_info_file = os.path.join(package_path, 'enet-cityscapes', 'enet-classes.txt')
    
    with open(class_info_file, 'r') as f:
        class_names = f.read().strip().split('\n')

    enet_neural_network = cv2.dnn.readNet(model_file)
    while not rospy.is_shutdown():
      if image_data is not None:
        input_frame = cv2.resize(image_data, (1024, 512))
    
        input_img_blob = cv2.dnn.blobFromImage(input_frame, 1.0 / 255, (1024, 512), 0, swapRB=True, crop=False)
        enet_neural_network.setInput(input_img_blob)
        enet_neural_network_output = enet_neural_network.forward()
    
        road_class_index = class_names.index('Road')  # 'Road' 클래스의 인덱스 가져오기
        road_class_map = np.argmax(enet_neural_network_output[0], axis=0) == road_class_index
    
        road_mask = np.zeros_like(input_frame)  # road_mask를 input_frame과 같은 크기로 생성
        road_mask[road_class_map] = input_frame[road_class_map]
    
        class_map = np.argmax(enet_neural_network_output[0], axis=0)
      
        if os.path.isfile('./enet-cityscapes/enet-colors.txt'):
          IMG_COLOR_LIST = (open('./enet-cityscapes/enet-colors.txt').read().strip().split("\n"))
          IMG_COLOR_LIST = [np.array(color.split(",")).astype("int") for color in IMG_COLOR_LIST]
          IMG_COLOR_LIST = np.array(IMG_COLOR_LIST, dtype="uint8")
        else:
          np.random.seed(1)
          IMG_COLOR_LIST = np.random.randint(0, 255, size=(len(class_names) - 1, 3), dtype="uint8")
          IMG_COLOR_LIST = np.vstack([[0, 0, 0], IMG_COLOR_LIST]).astype("uint8")
        
        class_map_mask = IMG_COLOR_LIST[class_map]
        class_map_mask = cv2.resize(class_map_mask, (input_frame.shape[1], input_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        enet_neural_network_output = ((0.60 * class_map_mask) + (0.40 * input_frame)).astype("uint8")
      
        class_legend = np.zeros(((len(class_names) * 25) + 25, 300, 3), dtype="uint8")
      
        for (i, (cl_name, cl_color)) in enumerate(zip(class_names, IMG_COLOR_LIST)):
          color_information = [int(color) for color in cl_color]
          cv2.putText(class_legend, cl_name, (5, (i * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
          cv2.rectangle(class_legend, (100, (i * 25)), (300, (i * 25) + 25), tuple(color_information), -1)
        
        combined_images = np.concatenate((input_frame, enet_neural_network_output), axis=1)
        
        contains_road = False
        
        road_class_index = class_names.index('Road')
        
        if road_class_index >= 0:
          road_class_mask = class_map == road_class_index
          if np.any(road_class_mask):
              contains_road = 1
        if distance < 100:
          cv2.putText(combined_images, f"Distance: {distance} meters", (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif 100<= distance < 300:
          cv2.putText(combined_images, f"Distance: {distance} meters", (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif 300<= distance < 1000:
          cv2.putText(combined_images, f"Distance: {distance} meters", (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else: cv2.putText(combined_images, f"Distance: {distance} meters", (1000, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Semantic Segmentation", road_mask)
        cv2.imshow('Results', combined_images)
        cv2.imshow("Class Legend", class_legend)        
                
        #cv2.imshow("Processed Image", image_data)
        key = cv2.waitKey(1)
        if key == ord('q'):
          cv2.destroyAllWindows()
          break
        
    rospy.spin()

