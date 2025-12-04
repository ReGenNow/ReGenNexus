"""
ReGenNexus Core - Jetson Orin Nano AI Example

This example demonstrates how to use ReGenNexus Core with a Jetson Orin Nano
to run AI inference with TensorRT acceleration and communicate results.

Requirements:
- NVIDIA Jetson Orin Nano
- Camera module (CSI or USB)
- ReGenNexus Core installed
- TensorRT and related dependencies
"""

import asyncio
import base64
import cv2
import numpy as np
import time
import json
import argparse
from datetime import datetime
import os
import threading

# ReGenNexus imports
# UAP_Client not available - this example needs updating for new API
from regennexus.core.message import Message
from regennexus.plugins.jetson import JetsonPlugin

# Check if TensorRT is available
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT not available. Will use OpenCV DNN instead.")

class JetsonOrinAIAgent:
    """Agent that runs AI inference on Jetson Orin Nano."""
    
    def __init__(self, entity_id, registry_url, camera_id=0, resolution=(1280, 720), 
                 model_path="models/resnet18.onnx", labels_path="models/imagenet_labels.txt"):
        """
        Initialize the Jetson Orin AI agent.
        
        Args:
            entity_id: Unique identifier for this agent
            registry_url: URL of the ReGenNexus registry
            camera_id: Camera device ID (0 for default)
            resolution: Camera resolution as (width, height)
            model_path: Path to the AI model file
            labels_path: Path to the labels file
        """
        self.entity_id = entity_id
        self.registry_url = registry_url
        self.camera_id = camera_id
        self.resolution = resolution
        self.model_path = model_path
        self.labels_path = labels_path
        
        self.client = None
        self.jetson_plugin = None
        self.camera = None
        self.model = None
        self.labels = []
        self.running = False
        self.inference_enabled = True
        
        # TensorRT/CUDA specific attributes
        self.trt_engine = None
        self.trt_context = None
        self.trt_bindings = []
        self.trt_inputs = []
        self.trt_outputs = []
        self.trt_stream = None
        
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
        
        # Load AI model
        self.load_model()
        
        # Load labels
        self.load_labels()
        
        print(f"Jetson Orin AI Agent initialized with ID: {self.entity_id}")
        print(f"TensorRT acceleration: {'Enabled' if TENSORRT_AVAILABLE else 'Disabled'}")
        
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
                f"format=NV12, framerate=30/1 ! "
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
                self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.camera.isOpened():
                raise RuntimeError("Failed to open camera")
                
            print(f"Camera initialized with resolution {self.resolution}")
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            raise
    
    def load_model(self):
        """Load the AI model."""
        try:
            if TENSORRT_AVAILABLE and os.path.exists(self.model_path):
                self._load_tensorrt_model()
            else:
                self._load_opencv_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_tensorrt_model(self):
        """Load model using TensorRT."""
        try:
            # Initialize TensorRT
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Load engine from file if it exists
            engine_path = self.model_path.replace('.onnx', '.engine')
            if os.path.exists(engine_path):
                print(f"Loading TensorRT engine from {engine_path}")
                with open(engine_path, 'rb') as f:
                    engine_data = f.read()
                
                runtime = trt.Runtime(TRT_LOGGER)
                self.trt_engine = runtime.deserialize_cuda_engine(engine_data)
            else:
                # Create engine from ONNX model
                print(f"Creating TensorRT engine from {self.model_path}")
                builder = trt.Builder(TRT_LOGGER)
                network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                parser = trt.OnnxParser(network, TRT_LOGGER)
                
                with open(self.model_path, 'rb') as model:
                    parser.parse(model.read())
                
                config = builder.create_builder_config()
                config.max_workspace_size = 1 << 30  # 1GB
                
                # Set FP16 mode if supported
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                
                self.trt_engine = builder.build_engine(network, config)
                
                # Save engine for future use
                with open(engine_path, 'wb') as f:
                    f.write(self.trt_engine.serialize())
            
            # Create execution context
            self.trt_context = self.trt_engine.create_execution_context()
            
            # Allocate memory for inputs and outputs
            for i in range(self.trt_engine.num_bindings):
                binding_dims = self.trt_engine.get_binding_shape(i)
                if self.trt_engine.binding_is_input(i):
                    # Input binding
                    self.trt_inputs.append(i)
                    input_size = trt.volume(binding_dims) * self.trt_engine.max_batch_size * np.dtype(np.float32).itemsize
                    device_input = cuda.mem_alloc(input_size)
                    self.trt_bindings.append(int(device_input))
                else:
                    # Output binding
                    self.trt_outputs.append(i)
                    output_size = trt.volume(binding_dims) * self.trt_engine.max_batch_size * np.dtype(np.float32).itemsize
                    device_output = cuda.mem_alloc(output_size)
                    self.trt_bindings.append(int(device_output))
            
            # Create CUDA stream
            self.trt_stream = cuda.Stream()
            
            print("TensorRT model loaded successfully")
            
        except Exception as e:
            print(f"Error loading TensorRT model: {e}")
            TENSORRT_AVAILABLE = False
            self._load_opencv_model()
    
    def _load_opencv_model(self):
        """Load model using OpenCV DNN."""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"Model file {self.model_path} not found. Using a dummy model.")
                self.model = None
                return
            
            # Load model with OpenCV DNN
            self.model = cv2.dnn.readNet(self.model_path)
            
            # Try to use CUDA backend if available
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            print("OpenCV DNN model loaded successfully")
            
        except Exception as e:
            print(f"Error loading OpenCV model: {e}")
            self.model = None
    
    def load_labels(self):
        """Load labels for classification."""
        try:
            if os.path.exists(self.labels_path):
                with open(self.labels_path, 'r') as f:
                    self.labels = [line.strip() for line in f.readlines()]
                print(f"Loaded {len(self.labels)} labels")
            else:
                print(f"Labels file {self.labels_path} not found. Using dummy labels.")
                self.labels = [f"Class_{i}" for i in range(1000)]
        except Exception as e:
            print(f"Error loading labels: {e}")
            self.labels = [f"Class_{i}" for i in range(1000)]
    
    async def handle_message(self, message):
        """
        Handle incoming messages.
        
        Args:
            message: UAP_Message object
        """
        print(f"Received message: {message.intent}")
        
        if message.intent == "capture_and_analyze":
            # Capture, analyze, and send an image
            await self.capture_analyze_and_send(message.sender)
            
        elif message.intent == "start_analysis_stream":
            # Start streaming analyzed images
            self.running = True
            recipient = message.sender
            interval = message.payload.get("interval", 1.0)  # Default: 1 second
            asyncio.create_task(self.stream_analyzed_images(recipient, interval))
            
        elif message.intent == "stop_stream":
            # Stop streaming images
            self.running = False
            
        elif message.intent == "enable_inference":
            # Enable AI inference
            self.inference_enabled = True
            
        elif message.intent == "disable_inference":
            # Disable AI inference
            self.inference_enabled = False
            
        elif message.intent == "set_resolution":
            # Set camera resolution
            width = message.payload.get("width", 1280)
            height = message.payload.get("height", 720)
            self.resolution = (width, height)
            
            # Reinitialize camera with new resolution
            if self.camera:
                self.camera.release()
            self.init_camera()
            
        elif message.intent == "load_model":
            # Load a new model
            model_path = message.payload.get("model_path")
            labels_path = message.payload.get("labels_path")
            
            if model_path:
                self.model_path = model_path
            
            if labels_path:
                self.labels_path = labels_path
                
            # Reload model and labels
            self.load_model()
            self.load_labels()
            
            # Send confirmation
            await self.client.send_message(UAP_Message(
                sender=self.entity_id,
                recipient=message.sender,
                intent="model_loaded",
                payload={
                    "model_path": self.model_path,
                    "labels_path": self.labels_path,
                    "success": True
                }
            ))
    
    async def capture_analyze_and_send(self, recipient):
        """
        Capture an image, run inference, and send results to the recipient.
        
        Args:
            recipient: Entity ID of the recipient
        """
        try:
            # Capture image
            ret, frame = self.camera.read()
            if not ret:
                raise RuntimeError("Failed to capture image")
            
            # Run inference if enabled
            results = None
            if self.inference_enabled:
                results = self.run_inference(frame)
            
            # Encode image as base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Create message with image data and inference results
            timestamp = datetime.now().isoformat()
            message = UAP_Message(
                sender=self.entity_id,
                recipient=recipient,
                intent="analysis_results",
                payload={
                    "image": img_base64,
                    "timestamp": timestamp,
                    "format": "jpg",
                    "resolution": {
                        "width": frame.shape[1],
                        "height": frame.shape[0]
                    },
                    "inference_results": results
                }
            )
            
            # Send message
            await self.client.send_message(message)
            print(f"Analysis results sent to {recipient}")
            
        except Exception as e:
            print(f"Error capturing and analyzing image: {e}")
            
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
    
    async def stream_analyzed_images(self, recipient, interval):
        """
        Stream analyzed images at regular intervals.
        
        Args:
            recipient: Entity ID of the recipient
            interval: Time between frames in seconds
        """
        print(f"Starting analysis stream to {recipient} at {interval}s intervals")
        
        while self.running:
            await self.capture_analyze_and_send(recipient)
            await asyncio.sleep(interval)
        
        print("Analysis stream stopped")
    
    def run_inference(self, frame):
        """
        Run AI inference on an image.
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            Dictionary of inference results
        """
        try:
            if TENSORRT_AVAILABLE and self.trt_engine:
                return self._run_tensorrt_inference(frame)
            elif self.model:
                return self._run_opencv_inference(frame)
            else:
                # Return dummy results if no model is available
                return {
                    "top_classes": [
                        {"label": "dummy_class_1", "confidence": 0.8},
                        {"label": "dummy_class_2", "confidence": 0.1},
                        {"label": "dummy_class_3", "confidence": 0.05}
                    ],
                    "inference_time": 0.01
                }
        except Exception as e:
            print(f"Error running inference: {e}")
            return {
                "error": str(e),
                "top_classes": []
            }
    
    def _run_tensorrt_inference(self, frame):
        """Run inference using TensorRT."""
        start_time = time.time()
        
        # Preprocess image
        input_size = (224, 224)  # Standard size for many models
        preprocessed = cv2.resize(frame, input_size)
        preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        preprocessed = preprocessed.astype(np.float32) / 255.0
        preprocessed = (preprocessed - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        preprocessed = np.transpose(preprocessed, (2, 0, 1))  # HWC to CHW
        preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
        
        # Copy input data to device
        cuda.memcpy_htod_async(self.trt_bindings[self.trt_inputs[0]], preprocessed, self.trt_stream)
        
        # Run inference
        self.trt_context.execute_async_v2(bindings=self.trt_bindings, stream_handle=self.trt_stream.handle)
        
        # Get output shape
        output_shape = self.trt_engine.get_binding_shape(self.trt_outputs[0])
        output_size = trt.volume(output_shape) * self.trt_engine.max_batch_size
        
        # Allocate output memory
        output = np.empty(output_size, dtype=np.float32)
        
        # Copy output data to host
        cuda.memcpy_dtoh_async(output, self.trt_bindings[self.trt_outputs[0]], self.trt_stream)
        
        # Synchronize stream
        self.trt_stream.synchronize()
        
        # Reshape output to match expected format
        output = output.reshape(output_shape)
        
        # Get top classes
        top_indices = np.argsort(output[0])[-5:][::-1]
        top_classes = [
            {
                "label": self.labels[idx] if idx < len(self.labels) else f"Class_{idx}",
                "confidence": float(output[0][idx])
            }
            for idx in top_indices
        ]
        
        inference_time = time.time() - start_time
        print(f"TensorRT inference time: {inference_time:.3f}s")
        
        return {
            "top_classes": top_classes,
            "inference_time": inference_time
        }
    
    def _run_opencv_inference(self, frame):
        """Run inference using OpenCV DNN."""
        start_time = time.time()
        
        # Preprocess image
        input_size = (224, 224)  # Standard size for many models
        blob = cv2.dnn.blobFromImage(
            frame, 1.0/255.0, input_size, 
            mean=[0.485, 0.456, 0.406], 
            swapRB=True, crop=False
        )
        
        # Set input and run inference
        self.model.setInput(blob)
        output = self.model.forward()
        
        # Get top classes
        top_indices = np.argsort(output[0])[-5:][::-1]
        top_classes = [
            {
                "label": self.labels[idx] if idx < len(self.labels) else f"Class_{idx}",
                "confidence": float(output[0][idx])
            }
            for idx in top_indices
        ]
        
        inference_time = time.time() - start_time
        print(f"OpenCV inference time: {inference_time:.3f}s")
        
        return {
            "top_classes": top_classes,
            "inference_time": inference_time
        }
    
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
    parser = argparse.ArgumentParser(description='Jetson Orin AI Agent')
    parser.add_argument('--entity-id', type=str, default='jetson_orin_ai',
                        help='Entity ID for this agent')
    parser.add_argument('--registry', type=str, default='localhost:8000',
                        help='URL of the ReGenNexus registry')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--width', type=int, default=1280,
                        help='Camera width')
    parser.add_argument('--height', type=int, default=720,
                        help='Camera height')
    parser.add_argument('--model', type=str, default='models/resnet18.onnx',
                        help='Path to the AI model file')
    parser.add_argument('--labels', type=str, default='models/imagenet_labels.txt',
                        help='Path to the labels file')
    
    args = parser.parse_args()
    
    # Create and run the agent
    agent = JetsonOrinAIAgent(
        entity_id=args.entity_id,
        registry_url=args.registry,
        camera_id=args.camera_id,
        resolution=(args.width, args.height),
        model_path=args.model,
        labels_path=args.labels
    )
    
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
