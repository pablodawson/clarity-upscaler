import cv2
import numpy as np
from ultralytics import YOLO
import random
import torch

class ImageSegmenter:
    def __init__(self, model_type="yolov8s-seg") -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO('models/'+ model_type +'.pt')
        self.model.to(self.device)

        self.is_show_bounding_boxes = True
        self.is_show_segmentation_boundary = False
        self.is_show_segmentation = False
        self.confidence_threshold = 0.5
        self.cls_clr = {}

        # params
        self.bb_thickness = 2
        self.bb_clr = (255, 0, 0)

        # variables
        self.masks = {}


    def get_cls_clr(self, cls_id):
        if cls_id in self.cls_clr:
            return self.cls_clr[cls_id]
        
        # gen rand color
        r = random.randint(50, 200)
        g = random.randint(50, 200)
        b = random.randint(50, 200)
        self.cls_clr[cls_id] = (r, g, b)
        return (r, g, b)

    def predict(self, image):            
        # params
        objects_data = [] 
        image = image.copy()
        predictions = self.model.predict(image)

        cls_ids = predictions[0].boxes.cls.cpu().numpy()
        bounding_boxes = predictions[0].boxes.xyxy.int().cpu().numpy()        
        cls_conf = predictions[0].boxes.conf.cpu().numpy()
        # segmentation
        if predictions[0].masks:
            seg_mask = predictions[0].masks.data.cpu().numpy()  
        else:
            seg_mask = np.array([])    
        
        for id, cls in enumerate(cls_ids):
            cls_clr = self.get_cls_clr(cls)

            # draw filled segmentation region
            if seg_mask.any() and  cls_conf[id] > self.confidence_threshold:

                self.masks[id] = seg_mask[id]
                
                if self.is_show_segmentation:
                    alpha = 0.8                

                    # converting the mask from 1 channel to 3 channels
                    colored_mask = np.expand_dims(seg_mask[id], 0).repeat(3, axis=0)
                    colored_mask = np.moveaxis(colored_mask, 0, -1)

                    # Resize the mask to match the image size, if necessary
                    if image.shape[:2] != seg_mask[id].shape[:2]:
                        colored_mask = cv2.resize(colored_mask, (image.shape[1], image.shape[0]))

                    # filling the mased area with class color
                    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=cls_clr)
                    image_overlay = masked.filled()                
                    image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

                # draw bounding box with class name and score
                if self.is_show_bounding_boxes and cls_conf[id] > self.confidence_threshold:
                    (x1, y1, x2, y2) = bounding_boxes[id]
                    cls_name = self.model.names[cls]
                    cls_confidence = cls_conf[id]
                    disp_str = cls_name +' '+ str(round(cls_confidence, 2))
                    cv2.rectangle(image, (x1, y1), (x2, y2), cls_clr, self.bb_thickness)
                    cv2.rectangle(image, (x1, y1), (x1+(len(disp_str)*9), y1+15), cls_clr, -1)
                    cv2.putText(image, disp_str, (x1+5, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                               
                # object variables
                (x1, y1, x2, y2) = bounding_boxes[id]
                center = x1+(x2-x1)//2, y1+(y2-y1)//2
                objects_data.append([cls, self.model.names[cls], center, self.masks[id], cls_clr])

        return bounding_boxes