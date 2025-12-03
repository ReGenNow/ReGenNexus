from regennexus.plugins.base import BasePlugin
import cv2
import numpy as np

class OpenCVPlugin(BasePlugin):
    def __init__(self, config=None):
        super().__init__(name="opencv", config=config)
        self.cameras = {}
        
    def initialize(self):
        # Initialize cameras or other OpenCV resources
        camera_indices = self.config.get("camera_indices", [0])
        for idx in camera_indices:
            self.cameras[idx] = cv2.VideoCapture(idx)
        return True
        
    def capture_frame(self, camera_idx=0):
        if camera_idx not in self.cameras:
            return None
        ret, frame = self.cameras[camera_idx].read()
        if ret:
            return frame
        return None
        
    def process_image(self, image, operation, **kwargs):
        # Implement various OpenCV operations
        if operation == "blur":
            return cv2.GaussianBlur(image, kwargs.get("kernel_size", (5,5)), 0)
        elif operation == "edge_detection":
            return cv2.Canny(image, kwargs.get("threshold1", 100), kwargs.get("threshold2", 200))
        # Add more operations as needed
        
    def cleanup(self):
        for camera in self.cameras.values():
            camera.release()
