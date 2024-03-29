# YOLOv3-Neural-Networks-project-Tensorflow 2.1.x
Final project for Neural Networks course -> Implementation of the paper "YOLOv3: An incremental improvement" (https://arxiv.org/pdf/1804.02767.pdf).

Full report [here](https://github.com/GianRomani/YOLOv3-Neural-Networks-project-/blob/main/NNs'%20Project%20Report%20-%20NN%2020_21%20-%20Romani%20Telinoiu%20.pdf).

### Prerequisites
This project is written in Python 3.6.9 using Tensorflow (deep learning), NumPy (numerical computing) and OpenCV (computer vision).

```
pip install -r requirements.txt
```

### Downloading official pretrained weights
Let's download official weights pretrained on COCO dataset. 

```
wget -P weights https://pjreddie.com/media/files/yolov3.weights
```

## Running the model
Now you can run the model using `detection.py` script by calling python3.

## Acknowledgments
* [Yolo v1 official paper](https://arxiv.org/pdf/1506.02640.pdf)
* [Yolo v2 official paper](https://arxiv.org/pdf/1612.08242v1.pdf)
* [Yolo v3 official paper](https://arxiv.org/pdf/1804.02767v1.pdf)
