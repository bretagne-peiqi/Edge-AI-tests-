# Distributed Yolo Model

## The Goal 
This project is used to study state of art about Inference Acceleration under Cloud-Edge Scenes
Where computation and memory resource could both be highly limited

## Method
The idea is to design a distributed model deployed in edge and cloud, the low level layers deployed 
at edge side takes input data and extracts feature representations in low dimension if possible. If
cann't finish it's inference only, it will send feature representations to cloud for the rest of work.

By doing this, we first reduce network bandwith, second, we can deploy models which could be too big 
for edge side. Finnally, with the help of reinforcement learning and scheduler, we could reschedule model
segmentation between edge side and cloud side, we cloud make priorities for important model at edge side 
and less important model at cloud side for inference. Even we could improve resouce utilization with the 
help of pipeline design.

## How it works

### First config models/yolo5s1.yaml yolo5s2.yaml for model slice, by executing python3 yolo.py, models will be put in weights dir.

### by executing python3 distributed-models.py, good weights in weights yolov5s.pt will be split and load in models slices generated in step one.  

### detect.py deployed only in edge side will call Model object which initialized with wei1.pt, yoloEdge.py will send output to yoloCloud.py.

### yoloCloud.py is the only component running on cloud side, it initialized with wei2.pt, listening on http server  

### for step 4,5; they need download their good model from a third party server.

## About this project
This project is based on official Yolov5 source code. 

## ToDo
Architecture and Code need to be improved, any contributions will be welcome
