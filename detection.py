from UTILS.boxes import *
from UTILS.load_weights import *
from LAYERS.darknet53 import *
from LAYERS.common_layers import *
from LAYERS.yolov3_model import *

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-05
LEAKY_RELU = 0.1
ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
WEIGHTS_PATH = '/home/cip/Desktop/NN Proj/[]REPO DI SUPPORTO/YOLOv3-Neural-Networks-project-/yolov3.weights'
NUM_CLASSES = 80
LABELS_PATH = '\DATASET\coco.names'
MODEL_IMG_SIZE = (416,416)
IOU_THRESHOLD = 0.5	
CONFIDENCE_THRESHOLD = 0.5
IMG_PATH = '/home/cip/Desktop/NN Proj/[]REPO DI SUPPORTO/YOLOv3-Neural-Networks-project-/DATASET/dog.jpg'

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


def main():
    yolo = yolov3(NUM_CLASSES, MODEL_IMG_SIZE, ANCHORS,
            IOU_THRESHOLD, CONFIDENCE_THRESHOLD, None, LEAKY_RELU)
    
    print("Loading weights...")
    print()
    load_yolo_weights(yolo, WEIGHTS_PATH)
    print("Loading done!\nCongrats!!!!!")
    print()

    '''
    img = cv2.imread(IMG_PATH)
    #print("@")
    #print(type(img))

    img = tf.image.decode_image(open(IMG_PATH, 'rb').read(), channels=3)
    img = tf.expand_dims(img,0)
    img = tf.image.resize(img, MODEL_IMG_SIZE) /255
    #print("!")
    #print(type(img))

    boxes, scores, classes, nums = yolo(img)

    print("Ispection of yolo(img)")
    print("-----------")
    print("Boxes-->")
    print()
    print(boxes)
    print("Score-->")
    print()
    print(scores)
    print("Classes-->")
    print()
    print(classes)
    print("Nums-->")
    print()
    print(nums)

    result = draw_boxes(img, (boxes, scores, classes, nums), CLASS_NAMES)
    #print(result)
    tf.keras.preprocessing.image.save_img('/home/cip/Desktop/NN Proj/[]REPO DI SUPPORTO/YOLOv3-Neural-Networks-project-/result.png',result[0])
    #cv2.imshow('image',result)
    '''
main()