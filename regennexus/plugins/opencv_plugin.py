"""
RegenNexus UAP - OpenCV Plugin

OpenCV-based computer vision plugin for camera capture and image processing.

Copyright (c) 2024-2025 ReGen Designs LLC
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Optional OpenCV import
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None
    np = None


class OpenCVPlugin:
    """
    OpenCV plugin for camera capture and image processing.

    This is a utility plugin for vision processing tasks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenCV plugin.

        Args:
            config: Configuration dict with optional 'camera_indices' list
        """
        self.config = config or {}
        self.cameras: Dict[int, Any] = {}
        self.initialized = False

        if not HAS_OPENCV:
            logger.warning("OpenCV not installed. Install with: pip install opencv-python")

    def initialize(self) -> bool:
        """
        Initialize cameras.

        Returns:
            True if initialization successful
        """
        if not HAS_OPENCV:
            logger.error("OpenCV not available")
            return False

        camera_indices = self.config.get("camera_indices", [0])
        for idx in camera_indices:
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    self.cameras[idx] = cap
                    logger.info(f"Camera {idx} initialized")
                else:
                    logger.warning(f"Could not open camera {idx}")
            except Exception as e:
                logger.error(f"Error initializing camera {idx}: {e}")

        self.initialized = len(self.cameras) > 0
        return self.initialized

    def capture_frame(self, camera_idx: int = 0) -> Optional[Any]:
        """
        Capture a frame from a camera.

        Args:
            camera_idx: Camera index to capture from

        Returns:
            numpy array of the frame, or None if failed
        """
        if not HAS_OPENCV:
            return None

        if camera_idx not in self.cameras:
            logger.warning(f"Camera {camera_idx} not initialized")
            return None

        ret, frame = self.cameras[camera_idx].read()
        if ret:
            return frame
        return None

    def process_image(self, image: Any, operation: str, **kwargs) -> Optional[Any]:
        """
        Process an image with OpenCV operations.

        Args:
            image: Input image (numpy array)
            operation: Operation name ('blur', 'edge_detection', 'grayscale', etc.)
            **kwargs: Operation-specific parameters

        Returns:
            Processed image, or None if operation failed
        """
        if not HAS_OPENCV or image is None:
            return None

        try:
            if operation == "blur":
                kernel_size = kwargs.get("kernel_size", (5, 5))
                return cv2.GaussianBlur(image, kernel_size, 0)

            elif operation == "edge_detection":
                threshold1 = kwargs.get("threshold1", 100)
                threshold2 = kwargs.get("threshold2", 200)
                return cv2.Canny(image, threshold1, threshold2)

            elif operation == "grayscale":
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            elif operation == "resize":
                width = kwargs.get("width", 640)
                height = kwargs.get("height", 480)
                return cv2.resize(image, (width, height))

            else:
                logger.warning(f"Unknown operation: {operation}")
                return image

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

    def cleanup(self) -> None:
        """Release all camera resources."""
        for idx, camera in self.cameras.items():
            try:
                camera.release()
                logger.info(f"Camera {idx} released")
            except Exception as e:
                logger.error(f"Error releasing camera {idx}: {e}")
        self.cameras.clear()
        self.initialized = False
