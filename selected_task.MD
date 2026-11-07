Drone-Based Object Detection
Task
Our project focuses on how object size affects detection performance in aerial imagery. Using the VisDrone dataset, we examine objects such as cars, pedestrians, and bicycles captured from different heights and distances.

In Phase I, we will develop a visualization to compare the model's predictions across different object sizes to identify patterns in detection errors.

In Phase II, we will explore the model's internal feature maps to understand how objects of different sizes are represented and investigate possible reasons for missed detections.

Dataset: https://github.com/VisDrone/VisDrone-Dataset

Model
We will use YOLO11n, an object detection model from Ultralytics, pretrained on the COCO dataset.

The model will be evaluated on selected VisDrone images, focusing on compatible object categories. We use the pretrained weights without additional training.

Repository: https://github.com/ultralytics/ultralytics

Pretrained checkpoint: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

Documentation: https://docs.ultralytics.com/models/yolo11
