# DLIP_LAB4

## project's objectives

## requirements
1. Hardware
NVDIA graphic cards
2. Software
* CUDA
* cuDNN
* Anaconda
* YOLO V5


    

Or You can follow the instructions from the yolov5 GitHub repository. [(requirements)](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)


## Follow the steps for setting in the Anaconda. 
> (Before starting, check your gpu to run)

    # create a conda env name=yolov5 (you can change your env name)
    conda create -n yolov5 python=3.8
    conda activate yolov5
    
    # Installation process
    conda install cudatoolkit=10.2
    conda install cudnn
    conda install tensorflow-gpu=2.3.0
    conda install keras
    conda install pillow
    conda install matplotlib
    conda install opencv
    
    # clone yolov5
    git clone https://github.com/ultralytics/yolov5.git
    
    # update
    conda update -yn base -c defaults conda
    
    # install Lib for YOLOv5
    conda install -c anaconda cython numpy pillow scipy seaborn pandas requests
    conda install -c conda-forge matplotlib pyyaml tensorboard tqdm opencv 

    # install pytorch
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    
    # Extra
    conda install -c conda-forge onnx
    
    
## essential codes


## how to run the program
