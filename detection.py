from UTILS.boxes import *
from UTILS.load_weights import *
from LAYERS.darknet53 import *
from LAYERS.common_layers import *
from LAYERS.yolov3_model import *
import cv2
import numpy as np
import colorsys
import random

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-05
LEAKY_RELU = 0.1
ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
#WEIGHTS_PATH = '/home/cip/Desktop/NN Proj/[]REPO DI SUPPORTO/YOLOv3-Neural-Networks-project-/yolov3.weights'
WEIGHTS_PATH = 'yolov3.weights'
NUM_CLASSES = 80
#LABELS_PATH = '/home/cip/Desktop/NN Proj/[]REPO DI SUPPORTO/YOLOv3-Neural-Networks-project-/DATASET/coco.names'
LABELS_PATH = 'DATASET/coco.names'
MODEL_IMG_SIZE = (416,416)
IOU_THRESHOLD = 0.5	
CONFIDENCE_THRESHOLD = 0.5
#IMG_PATH  = '/home/cip/Desktop/NN Proj/[]REPO DI SUPPORTO/YOLOv3-Neural-Networks-project-/DATASET/dog.jpg'
IMG_PATH  = 'DATASET/dog.jpg'
#PRED_PATH = '/home/cip/Desktop/NN Proj/[]REPO DI SUPPORTO/YOLOv3-Neural-Networks-project-/DATASET' 
PRED_PATH = 'DATASET' 

CLASS_NAMES =  ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
  "banana","apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
  "cake","chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
  "mouse","remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator","book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def detect_image(Yolo, image_path, output_path, input_size=416, show=False, CLASSES=LABELS_PATH, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    original_image      = cv2.imread(IMG_PATH) #load image to be detected as np.array using CV2

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size]) #resize and pad original_img
    image_data = image_data[np.newaxis, ...].astype(np.float32)                      #add 1D to processed_img

    pred_bbox = Yolo.predict(image_data)  #using tf.keras.Model.predict() based on model built
    
    pred_bbox = tf.reshape(pred_bbox[0], (-1, tf.shape(pred_bbox[0])[-1])) #reshaping into 2D 

    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    # print("%%%%%%%PostProcess%%%%%%%%")
    # print(bboxes)
    # print(tf.shape(bboxes))

    #bboxes = nms(bboxes, iou_threshold, method='nms')
    bboxes = non_max_suppression(bboxes, 80, 10, 0.5, 0.5)
    image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
    # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))

    #if output_path != '': cv2.imwrite(output_path, image)
    if show:
        # Show the image
        cv2.imshow("predicted image", image)
        # Load and hold the image
        cv2.waitKey(0)
        # To close the window after the required kill value was provided
        cv2.destroyAllWindows()
        
    return image


def main():
    yolo = yolov3(NUM_CLASSES, MODEL_IMG_SIZE, ANCHORS,
            IOU_THRESHOLD, CONFIDENCE_THRESHOLD, None, LEAKY_RELU)
    
    print("Loading weights...")
    print()
    load_yolo_weights(yolo, WEIGHTS_PATH)
    print("Loading done!\nCongrats!!!!!")
    print()
    
    detect_image(yolo, IMG_PATH, PRED_PATH, 416, show=True, rectangle_colors=(255,0,0))

main()