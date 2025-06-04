# RipCurrentDetector

Using computer vision through drones perspective, YOLOV8 object detection, CVAT, and UCSC rip current datset; developed a fully functional prototype that detects rip currents in real time. The UCSC data labels were not compatible with YoloV8 so I developed a C++ and Python program to convert given bounded box data labels to YOLOv8 format. Trained, validated, and tested YOLOv8 object detection algorithm to confidently detect rip currents from a drone’s perspective with up to 92% accuracy. 
