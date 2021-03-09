import cv2
import numpy as np
import tensorflow as tf
import colorsys
import random

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-05
LEAKY_RELU = 0.1
ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
WEIGHTS_PATH = '/home/cip/Desktop/NN Proj/[]REPO DI SUPPORTO/YOLOv3-Neural-Networks-project-/yolov3.weights'
NUM_CLASSES = 80
LABELS_PATH = '/home/cip/Desktop/NN Proj/[]REPO DI SUPPORTO/YOLOv3-Neural-Networks-project-/DATASET/coco.names'
MODEL_IMG_SIZE = (416,416)
IOU_THRESHOLD = 0.5	
CONFIDENCE_THRESHOLD = 0.5
IMG_PATH  = '/home/cip/Desktop/NN Proj/[]REPO DI SUPPORTO/YOLOv3-Neural-Networks-project-/DATASET/dog.jpg'
PRED_PATH = '/home/cip/Desktop/NN Proj/[]REPO DI SUPPORTO/YOLOv3-Neural-Networks-project-/DATASET' 

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


def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def image_preprocess(image, target_size):
    
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    '''
    cv2.imwrite('color_img.jpg', image_resized)
    cv2.imshow("image", image_resized)
    cv2.waitKey()
    '''

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    '''
    cv2.imwrite('color_img2.jpg', image_paded)
    cv2.imshow("image", image_paded)
    cv2.waitKey()
    '''

    return image_paded

def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,              #(pred_xy - (pred_wh/2), pred_xy + (pred_wh/2)) -->
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)    # getting bottom left top right coord of pred_bbox
    print("inspecting shape")
    print(np.shape(pred_coor))
    print(pred_coor[1])

    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)
    print()
    print("Printing resize ratio")
    print(resize_ratio)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    print()
    print("Printing dw dh")
    print(dw)
    print(dh)

    print()
    print("Printing pred_coor[:, 0::2]")
    print(pred_coor[:, 0::2])
    print()
    print("Printing pred_coor[:, 1::2]")
    print(pred_coor[:, 1::2])


    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    print()
    print()
    print("!!!!!!!!!!!!!!!!!!")
    print(bboxes)
    print(tf.shape(bboxes))
    print("!!!!!!!!!!!!!!!!!!")

    '''             0       1       2       3
    bbox input = [ x_min, y_min, x_max, y_max, score, class ]
                    1       0       3       2

    Bounding boxes are supplied as [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any diagonal pair of box corners 
    and the coordinates can be provided as normalized (i.e., lying in the interval [0, 1]) or absolute.

    boxes 	A 2-D float Tensor of shape [num_boxes, 4].
    scores 	A 1-D float Tensor of shape [num_boxes] representing a single score corresponding to each box (each row of boxes).
    '''
    
    boxes, score, clss = tf.split(bboxes, [4,1,1], axis=-1)   

    score = tf.reshape(score, tf.shape(score)[0])

    boxes = tf.dtypes.cast(boxes, tf.float32)
    score = tf.dtypes.cast(score, tf.float32)

    print("testing split")
    print(boxes)
    print(type(boxes))
    print(score)
    print(type(score))
    print(clss)
    print(type(clss))

    selected_indices = tf.image.non_max_suppression(boxes, score, 15, 0.4, 0.4)


    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0 
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # Process 3: Calculate this bounding box A and
            # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold 
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    print("Inspecting result on nms tf.image")
    print("-->")

    print("Selected Indices:")
    print(selected_indices)
    print(tf.shape(selected_indices))
    
    selected_boxes = tf.gather(boxes, selected_indices)

    print()
    print("Selected boxes:")
    print(selected_boxes)
    print(tf.shape(selected_boxes))

    print()
    print("best_bboxes of manual NMS:")
    print(best_bboxes)
    print(tf.shape(best_bboxes))
    print()
    print()
    
    return best_bboxes

def draw_bbox(image, bboxes, CLASSES=LABELS_PATH, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=False):   
    print("!!!!!!!!!!")
    print(bboxes)

    
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    #print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " "+str(score)

            try:
                label = "{}".format(NUM_CLASS[class_ind]) + score_str
            except KeyError:
                print("You received KeyError, this might be that you are trying to use yolo original weights")
                print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image

def draw_boxes(img, outputs, class_names): #visual representation of model's computed BB
    
    '''
    print()
    print("Ispection of input var 'output' of draw_bb")
    print(outputs)
    print()
    '''

    nms_boxes, nms_score, nms_classes, nums = outputs

    boxes, score, classes, nums = nms_boxes[0], nms_score[0], nms_classes[0], nums[0] #values returned explicitely by the yolo detection

    '''
    print("Ispection of outputs[0] variable")
    print("-----------")
    print("Boxes-->")
    print()
    print(boxes)
    print("Score-->")
    print()
    print(score)
    print("Classes-->")
    print()
    print(classes)
    print("Nums-->")
    print()
    print(nums)
    '''

    img_scale = np.flip(img.shape[0:2]) #get widht/height of the image to adjust bb dim

    for i in range(nums): #for every detection
        x1y1 = tuple((np.array(boxes[i][0:2])*img_scale).astype(np.int32)) #rescaled bb topleft_coordinate
        x2y2 = tuple((np.array(boxes[i][2:4])*img_scale).astype(np.int32)) #rescaled bb bottomright_coordinate

        img = cv2.rectangle(img, x1y1, x2y2, (255,0,0), 2) #draw rectangle bases on new coordinates
        img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], score[i]),txty, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2) #for each detection complete the rectangle description
    #print("#")
    #print(type(img))
    return img

def build_boxes(inputs): #compute topleft and bottom right coordinatas of the BB
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #print(inputs)

    #retrieve information after all detections have been performed during model definition,
    #for position of specific values refer to yolo_layer function's output in LAYERS dir
    tx, ty, tw, th, confidence, classes = \
        tf.split(inputs, [1,1,1,1,1,-1], axis=-1)
    
    #now we simply compute the coordinates of the BB
    x1 = tx - tw / 2     #topleft_x corner
    y1 = ty - th / 2    #topleft_y corner 

    x2 = tx + tw / 2     #bottomleft_x corner
    y2 = ty + th / 2    #bottomleft_y corner

    #print("@@@@@@@@@@@@@@@@@")
    #print(x1,y1,x2,y2)

    #finally we pack all values together to be used for NMS
    boxes = tf.concat([y1, x1, 
                        y2, x2, 
                        confidence, classes], axis=-1)


    return boxes

def nms_2 (inputs, classes, iou_threshold, confidence_threshold):

    #retrive information from output of build_boxes in a compatible format with NMS function
    diag_coord, confidence, classes = \
        tf.split(inputs, [4,1,-1], axis = -1)
    
    #combine confidence scores with all classes
    scores = confidence * classes

    #tf function specification
    #tf.image.combined_non_max_suppression(
        #boxes, scores, 
        #max_output_size_per_class, 
        #max_total_size, 
        #iou_threshold=0.5,
        #score_threshold=float('-inf')
    #)

    nms_boxes, nms_scores, nms_classes, valid_detections = \
        tf.image.combined_non_max_suppression(
            boxes = tf.reshape(diag_coord, (tf.shape(diag_coord)[1], -1, 1, 4)),
            scores = tf.reshape(scores, (tf.shape(scores)[1], -1, tf.shape(classes)[-1])),
            max_output_size_per_class = 5,
            max_total_size = 5,
            iou_threshold = iou_threshold,
            score_threshold = confidence_threshold
        )
    '''
    print("Inspecting result of NMS")
    print(nms_boxes)
    print(nms_scores)
    print(nms_classes)
    print(valid_detections)
    '''

    return nms_boxes, nms_scores, nms_classes, valid_detections
    