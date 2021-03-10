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
    '''
    print("inspecting shape")
    print(np.shape(pred_coor))
    print(pred_coor[1])
    '''
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)
    # print()
    # print("Printing resize ratio")
    # print(resize_ratio)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    # print()
    # print("Printing dw dh")
    # print(dw)
    # print(dh)

    # print()
    # print("Printing pred_coor[:, 0::2]")
    # print(pred_coor[:, 0::2])
    # print()
    # print("Printing pred_coor[:, 1::2]")
    # print(pred_coor[:, 1::2])


    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1)) #multiply is applied only to axis=-1 thanks to reduce
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1) #newaxis = None


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

    '''
                    0       1       2       3
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
    print(tf.shape(boxes))
    print(score)
    print(tf.shape(score))
    print(clss)
    print(tf.shape(clss))

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


def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold,
                        confidence_threshold):
    """Performs non-max suppression separately for each class.
    Args:
        inputs: Tensor input.
        n_classes: Number of classes.
        max_output_size: Max number of boxes to be selected for each class.
        iou_threshold: Threshold for the IOU.
        confidence_threshold: Threshold for the confidence score.
    Returns:
        A list containing class-to-boxes dictionaries
            for each sample in the batch.
    """
    print('------NMS-------')
    print()
    print(f'='*30)
    print(inputs)
    print(tf.shape(inputs))

    classes = tf.reshape(inputs[:, 5:] , (-1))
    print(f'='*30)
    print(classes)
    print(tf.shape(classes))

    classes = tf.expand_dims(tf.cast(classes, dtype=tf.float32), axis=-1)

    boxes = tf.concat([inputs[:, :5], classes], axis=-1)
    print(f'='*30)
    print(boxes)
    print(tf.shape(boxes))
    
    array_bboxes = []

    for cls in range(n_classes):
        #print("Currently analizing class number: "+str(cls))
        mask = tf.equal(boxes[:, 5], cls)
        mask_shape = mask.get_shape()

        if mask_shape.ndims != 0:
            class_boxes = tf.boolean_mask(boxes, mask)
            ''' 
            print(f'+'*50)
            print('Class_boxes before NMS')
            print(class_boxes)
            '''
            
            boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes, [4, 1, -1], axis=-1)
            boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
            indices = tf.image.non_max_suppression(boxes_coords, boxes_conf_scores, max_output_size, iou_threshold)
            
            class_boxes = tf.gather(class_boxes, indices)
            if tf.shape(class_boxes)[0] != 0 :
                '''
                print(f'!'*50)
                print('Class_boxes')
                print(class_boxes)
                print(f'Class:{class_boxes[:,5]}, shape: {tf.shape(class_boxes)[0]}')
                '''
                array_bboxes.append(class_boxes)
    

    best_bboxes = tf.concat(array_bboxes, axis=0)

    return best_bboxes

def draw_boxes(img, outputs, class_names): #visual representation of model's computed BB

    classes = read_class_names(class_names)
    print(classes)
    
    for detection in outputs:
        x1y1, x2y2, score, clss = tf.split(detection, [2,2,1,1], axis=0)

        x1y1    = (int(x1y1[0]),int(x1y1[1]))
        x2y2    = (int(x2y2[0]),int(x2y2[1]))
        score   = float(score[0])
        idx     = int(clss[0])
        
        class_label = classes[idx]

        img = cv2.rectangle(img, x1y1, x2y2, (255,0,0), 2) #draw rectangle bases on new coordinates
        img = cv2.putText(img, '{} {:.4f}'.format(class_label, score),x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2) #for each detection complete the rectangle description
        
    return img

