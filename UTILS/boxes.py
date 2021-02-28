import cv2
import numpy as np



def draw_boxes(img, outputs, class_names): #visual representation of model's computed BB
    boxes, score, classes, nums = outputs 
    boxes, score, classes, nums = boxes[0], score[0], classes[0], nums[0] #values returned explicitely by the yolo detection

    img_scale = np.flip(img.shape[0:2]) #get widht/height of the image to adjust bb dim

    for i in range(nums): #for every detection
        txty = tuple((np.array(boxes[i][0:2])*img_scale).astype(np.int32)) #rescaled bb topleft_coordinate
        twth = tuple((np.array(boxes[i][2:4])*img_scale).astype(np.int32)) #rescaled bb bottomright_coordinate

        img = cv2.rectangle(img, txty, twth, (255,0,0), 2) #draw rectangle bases on new coordinates
        img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], score[i]),txty, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2) #for each detection complete the rectangle description

    return img

def build_boxes(inputs): #compute topleft and bottom right coordinatas of the BB
    