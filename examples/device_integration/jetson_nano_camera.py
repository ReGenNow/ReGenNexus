"""
ReGenNexus Core - Jetson Nano Camera Example

This example demonstrates how to use ReGenNexus Core with a Jetson Nano
to capture camera images and process them with basic CUDA acceleration.

Requirements:
- NVIDIA Jetson Nano
- Camera module (CSI or USB)
- ReGenNexus Core installed
"""

import asyncio
import base64
import cv2
import numpy as np
import time
import json
import argparse
from datetime import datetime

# ReGenNexus imports
# UAP_Client not available - this example needs updating for new API
from regennexus.core.message import Message
from regennexus.plugins.jetson import JetsonPlugin

# Check if CUDA is available for OpenCV
CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0

class JetsonNanoCameraAgent:
    """Agent that captures and processes camera images on Jetson Nano."""
    
    def __init__(self, entity_id, registry_url, camera_id=0, resolution=(640, 480), fps=30):
        """
        Initialize the Jetson Nano camera agent.
        
        Args:
            entity_id: Unique identifier for this agent
            registry_url: URL of the ReGenNexus registry
            camera_id: Camera device ID (0 for default)
            resolution: Camera resolution as (width, height)
            fps: Frames per second
        """
        self.entity_id = entity_id
        self.registry_url = registry_url
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        
        self.client = None
        self.jetson_plugin = None
        self.camera = None
        self.running = False
        self.processing_enabled = False
        
    async def initialize(self):
        """Initialize the agent and connect to the registry."""
        # Initialize ReGenNexus client
        self.client = UAP_Client(entity_id=self.entity_id, registry_url=self.registry_url)
        await self.client.connect()
        
        # Register message handler
        self.client.register_message_handler(self.handle_message)
        
        # Initialize Jetson plugin
        self.jetson_plugin = JetsonPlugin()
        await self.jetson_plugin.initialize()
        
        # Initialize camera
        self.init_camera()
        
        print(f"Jetson Nano Camera Agent initialized with ID: {self.entity_id}")
        print(f"CUDA acceleration: {'Enabled' if CUDA_AVAILABLE else 'Disabled'}")
        
        # Send device info to registry
        device_info = await self.jetson_plugin.get_device_info()
        await self.client.send_message(UAP_Message(
            sender=self.entity_id,
            recipient="registry",
            intent="device_info",
            payload=device_info
        ))
    
    def init_camera(self):
        """Initialize the camera."""
        try:
            # Try to use GStreamer pipeline for CSI camera
            gst_str = (
                f"nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM), width={self.resolution[0]}, height={self.resolution[1]}, "
                f"format=NV12, framerate={self.fps}/1 ! "
                f"nvvidconv ! video/x-raw, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink"
            )
            
            self.camera = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            
            # If GStreamer pipeline fails, fall back to regular camera
            if not self.camera.isOpened():
                print("CSI camera not available, falling back to USB camera")
                self.camera = cv2.VideoCapture(self.camera_id)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.camera.isOpened():
                raise RuntimeError("Failed to open camera")
                
            print(f"Camera initialized with resolution {self.resolution}")
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            raise
    
    async def handle_message(self, message):
        """
        Handle incoming messages.
        
        Args:
            message: UAP_Message object
        """
        print(f"Received message: {message.intent}")
        
        if message.intent == "capture_image":
            # Capture and send an image
            await self.capture_and_send_image(message.sender)
            
        elif message.intent == "start_stream":
            # Start streaming images
            self.running = True
            recipient = message.sender
            interval = message.payload.get("interval", 1.0)  # Default: 1 second
            asyncio.create_task(self.stream_images(recipient, interval))
            
        elif message.intent == "stop_stream":
            # Stop streaming images
            self.running = False
            
        elif message.intent == "enable_processing":
            # Enable image processing
            self.processing_enabled = True
            
        elif message.intent == "disable_processing":
            # Disable image processing
            self.processing_enabled = False
            
        elif message.intent == "set_resolution":
            # Set camera resolution
            width = message.payload.get("width", 640)
            height = message.payload.get("height", 480)
            self.resolution = (width, height)
            
            # Reinitialize camera with new resolution
            if self.camera:
                self.camera.release()
            self.init_camera()
    
    async def capture_and_send_image(self, recipient):
        """
        Capture an image and send it to the recipient.
        
        Args:
            recipient: Entity ID of the recipient
        """
        try:
            # Capture image
            ret, frame = self.camera.read()
            if not ret:
                raise RuntimeError("Failed to capture image")
            
            # Process image if enabled
            if self.processing_enabled:
                frame = self.process_image(frame)
            
            # Encode image as base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Create message with image data
            timestamp = datetime.now().isoformat()
            message = UAP_Message(
                sender=self.entity_id,
                recipient=recipient,
                intent="image_data",
                payload={
                    "image": img_base64,
                    "timestamp": timestamp,
                    "format": "jpg",
                    "resolution": {
                        "width": frame.shape[1],
                        "height": frame.shape[0]
                    }
                }
            )
            
            # Send message
            await self.client.send_message(message)
            print(f"Image sent to {recipient}")
            
        except Exception as e:
            print(f"Error capturing image: {e}")
            
            # Send error message
            error_message = UAP_Message(
                sender=self.entity_id,
                recipient=recipient,
                intent="error",
                payload={
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
            await self.client.send_message(error_message)
    
    async def stream_images(self, recipient, interval):
        """
        Stream images at regular intervals.
        
        Args:
            recipient: Entity ID of the recipient
            interval: Time between frames in seconds
        """
        print(f"Starting image stream to {recipient} at {interval}s intervals")
        
        while self.running:
            await self.capture_and_send_image(recipient)
            await asyncio.sleep(interval)
        
        print("Image stream stopped")
    
    def process_image(self, frame):
        """
        Process an image with CUDA acceleration if available.
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            Processed image frame
        """
        try:
            if CUDA_AVAILABLE:
                # Use CUDA for image processing
                start_time = time.time()
                
                # Upload image to GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # Apply Gaussian blur
                gpu_frame = cv2.cuda.createGaussianFilter(
                    cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 1.0
                ).apply(gpu_frame)
                
                # Apply Canny edge detection
                gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                gpu_edges = cv2.cuda.createCannyEdgeDetector(100, 200).detect(gpu_gray)
                
                # Download result from GPU
                edges = gpu_edges.download()
                
                # Convert back to color for visualization
                edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                
                # Combine original and edges
                result = cv2.addWeighted(frame, 0.7, edges_color, 0.3, 0)
                
                processing_time = time.time() - start_time
                print(f"CUDA processing time: {processing_time:.3f}s")
                
                return result
            else:
                # Use CPU for image processing
                start_time = time.time()
                
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(frame, (5, 5), 1.0)
                
                # Apply Canny edge detection
                gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                
                # Convert back to color for visualization
                edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                
                # Combine original and edges
                result = cv2.addWeighted(frame, 0.7, edges_color, 0.3, 0)
                
                processing_time = time.time() - start_time
                print(f"CPU processing time: {processing_time:.3f}s")
                
                return result
                
        except Exception as e:
            print(f"Error processing image: {e}")
            return frame
    
    async def run(self):
        """Run the agent."""
        try:
            await self.initialize()
            
            # Keep the agent running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            # Clean up resources
            if self.camera:
                self.camera.release()
            
            if self.client:
                await self.client.disconnect()

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Jetson Nano Camera Agent')
    parser.add_argument('--entity-id', type=str, default='jetson_nano_camera',
                        help='Entity ID for this agent')
    parser.add_argument('--registry', type=str, default='localhost:8000',
                        help='URL of the ReGenNexus registry')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--width', type=int, default=640,
                        help='Camera width')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera height')
    parser.add_argument('--fps', type=int, default=30,
                        help='Camera FPS')
    
    args = parser.parse_args()
    
    # Create and run the agent
    agent = JetsonNanoCameraAgent(
        entity_id=args.entity_id,
        registry_url=args.registry,
        camera_id=args.camera_id,
        resolution=(args.width, args.height),
        fps=args.fps
    )
    
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
