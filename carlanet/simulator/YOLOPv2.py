import torch
import numpy as np
import os
import cv2
from ..utils.yolopv2 import letterbox, lane_line_mask, driving_area_mask, show_seg_result
from ..utils.path import get_project_root

class YOLOPv2:
    """
    YOLOPv2 is a model for doing inference on 3 tasks simultaneously: object detection, drivable area detection, and lane detection.

    Author: Cheng Han, Qichao Zhao, Shuyi Zhang, Yinzi Chen, Zhenlin Zhang, Jinwei Yuan
    Paper: https://arxiv.org/abs/2208.11434
    Implementation code: https://github.com/CAIC-AD/YOLOPv2.git
    """
    def __init__(self, weights_path: str=None, device: str = "cpu"):
        """
        Initializes the YOLOPv2 model.

        Args:
            weights_path (str): The path to the weights file.
            device (str, optional): The device to run the model on. Defaults to "cpu".
        """
        self.device = device
        self.model = None

        # Load model
        weights_path = weights_path or os.path.join(get_project_root(), "model", "yolopv2.pt")
        self.model = torch.jit.load(weights_path, map_location=self.device)

        if self.device == "cuda":
            self.model.half() # Use FP16 for CUDA inference for faster inference

        # Set model to eval mode
        self.model.eval()

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Performs inference on the given image for the 3 tasks: object detection, drivable area detection, and lane detection.

        Args:
            image (np.ndarray): The image to perform inference on.

        Returns:
            np.ndarray: The output of the model.
        """
        # Transform image
        img, real_img = self.transform_image(image) 

        img = torch.from_numpy(img).to(self.device) # Convert to tensor

        img = img.half() if self.device == "cuda" else img.float() # Use FP16 for CUDA inference for faster inference
        img /= 255.0 # Normalize image

        if img.ndimension() == 3: # If image is 3x640x640, add a batch dimension
            img = img.unsqueeze(0)
        
        # Perform inference
        [preds, anchors], segmentations, lane_detection = self.model(img)

        return [preds, anchors], segmentations, lane_detection, real_img
    
    def show_detection(self, predictions):
        """
        Detects objects, drivable area, and lane lines from the given predictions.

        Args:
            predictions (np.ndarray): The predictions from the model.

        Returns:
            np.ndarray: The detected objects, drivable area, and lane lines.
        """
        # Get predictions
        [preds, anchors], segmentations, lane_detection, r_img = predictions

        lane_mask = lane_line_mask(lane_detection)
        d_area_mask = driving_area_mask(segmentations)

        mask = show_seg_result(r_img, (d_area_mask, lane_mask), is_demo=True)

        # Define a structuring element for erosion
        kernel = np.ones((5,5),np.uint8)

        # Apply erosion
        eroded_mask = cv2.erode(mask, kernel, iterations=1)

        # Smoothen the eroded mask using Gaussian blur
        smooth_mask = cv2.GaussianBlur(eroded_mask, (5, 5), 0)

        return smooth_mask

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """
        It transforms the given image into a 3x640x640 tensor with letterbox.

        Args:
            image (np.ndarray): The image to transform.

        Returns:
            np.ndarray: The transformed image.
        """
        # Resize image into 1280x720 with INTER_LINEAR interpolation.
        image_720 = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)

        # Use letterbox to resize image into 640x640.
        img = letterbox(image_720, new_shape=640, stride=32)[0]

        # Convert BGR to RGB and to 3x640x640 tensor. ascontiguousarray() is needed to ensure the tensor is stored in a contiguous chunk of memory.
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return img, image_720