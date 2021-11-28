import cv2
import torch

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


class YoloDetector():
    def __init__(self, weights='./best1.pt', img_size=480, conf_thres=0.4, iou_thres=0.4):
        self.device = select_device('0')
    
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        img_size = check_img_size(img_size, s=self.model.stride.max())  # check img_size

        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        #half = device.type != 'cpu'  # half precision only supported on CUDA
        #self.half = half
        #if half:
        #    model.half()  # to FP16



    def detect(self, image_original):

        image = cv2.resize(image_original, (self.img_size, self.img_size))
        # Run inference
        img = torch.from_numpy(image).to(self.device)
        #img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img =  img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.permute((2, 0, 1))
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
    
        # Inference
        with torch.no_grad():
            pred = self.model(img)[0]
    
        # Apply NMS
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=[0,1])  # select only dog (class 16)

        # Process detections
        height_ratio = image_original.shape[0] / self.img_size
        width_ratio = image_original.shape[1] / self.img_size
        if det is not None and len(det) and det[0] is not None:
            det = det[0]  # we have only one frame, so select it
            # Rescale boxes from img_size to im0 size
            det[:, 0] *= width_ratio
            det[:, 1] *= height_ratio
            det[:, 2] *= width_ratio
            det[:, 3] *= height_ratio
            return det.cpu().numpy()
        else:
            return []

