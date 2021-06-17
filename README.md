# DLIP_LAB4

## Project's objectives

## Requirements
1. Hardware
NVDIA graphic cards
2. Software
> * CUDA
> * cuDNN
> * Anaconda
> * YOLO V5


    

Or You can follow the instructions from the yolov5 GitHub repository. [(requirements)](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)


### Follow the steps for setting in the Anaconda. 
>   Before starting, check your gpu to match the version
>   
>       # create a conda env name=yolov5 (you can change your env name)
>       conda create -n yolov5 python=3.8
>       conda activate yolov5
>       
>       # Installation process
>       conda install cudatoolkit=10.2
>       conda install cudnn
>       conda install tensorflow-gpu=2.3.0
>       conda install keras
>       conda install pillow
>       conda install matplotlib
>       conda install opencv
>       
>       # clone yolov5
>       git clone https://github.com/ultralytics/yolov5.git
>       
>       # update
>       conda update -yn base -c defaults conda
>       
>       # install Lib for YOLOv5
>       conda install -c anaconda cython numpy pillow scipy seaborn pandas requests
>       conda install -c conda-forge matplotlib pyyaml tensorboard tqdm opencv 
>   
>       # install pytorch
>       conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
>      
>       # Extra
>       conda install -c conda-forge onnx
    
## Essential codes to understand the program

>   Editing Parser
>   Since we are using 'YOLO V5s model', we set the default for weights as 'yolov5s'.
>   
>       parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')

>   Image size is set as 608.
>   
>       parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')

>   conf-thres value increased to 0.3
>       parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')

>   iou-thres value decreased to 0.4
>   
>       parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')

>   view-img's action is set as 'store_false' to view the image. 
>   
>       parser.add_argument('--view-img', action='store_false', help='display results')
    
>   save-txt's action is set as 'store_false' to save the result.
>   
>       parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')

>   classes's default is added as 2 to only view car class.
>   
>       parser.add_argument('--classes', nargs='+', type=int, default=2, help='filter by class: --class 0, or --class 0 2 3')

>   line-thickness is edited to 2.
>   
>       parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')






## How to run the program
