import cv2
import numpy as np
import tensorflow as tf



def draw_boxes(img, outputs, class_names): #visual representation of model's computed BB
    nms_boxes, nms_score, nms_classes, nums = outputs 
    boxes, score, classes, nums = nms_boxes[0], nms_score[0], nms_classes[0], nums[0] #values returned explicitely by the yolo detection

    img_scale = np.flip(img.shape[0:2]) #get widht/height of the image to adjust bb dim

    for i in range(nums): #for every detection
        x1y1 = tuple((np.array(boxes[i][0:2])*img_scale).astype(np.int32)) #rescaled bb topleft_coordinate
        x2y2 = tuple((np.array(boxes[i][2:4])*img_scale).astype(np.int32)) #rescaled bb bottomright_coordinate

        img = cv2.rectangle(img, x1y1, x2y2, (255,0,0), 2) #draw rectangle bases on new coordinates
        img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], score[i]),txty, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2) #for each detection complete the rectangle description

    return img

def build_boxes(inputs): #compute topleft and bottom right coordinatas of the BB
    
    #retrieve information after all detections have been performed during model definition,
    #for position of specific values refer to yolo_layer function's output in LAYERS dir
    tx, ty, tw, th, confidence, classes = \
        tf.split(inputs, [1,1,1,1,1,-1], axis=-1)
    
    #now we simply compute the coordinates of the BB
    x1 = tx - tw / 2     #topleft_x corner
    y1 = ty - th / 2    #topleft_y corner 

    x2 = tx + tw / 2     #bottomleft_x corner
    y2 = ty + th / 2    #bottomleft_y corner

    #finally we pack all values together to be used for NMS
    boxes = tf.concat([y1, x1, 
                        y2, x2, 
                        confidence, classes], axis=-1)


    return boxes

def nms (inputs, classes, iou_threshold, confidence_threshold):
    
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
            max_output_size_per_class = 100,
            max_total_size = 100,
            iou_threshold = iou_threshold,
            score_threshold = confidence_threshold
        )

    return nms_boxes, nms_scores, nms_classes, valid_detections
    