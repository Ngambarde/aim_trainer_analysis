import torch
from ultralytics import YOLO
import os

# -- Configuration ---
# Path to your trained YOLOv8 segmentation .pt model
PT_MODEL_PATH = r"models/best.pt"

# Input image size for the TensorRT engine
# Use the size you trained with, or a standard size like 640 or 1280.
# Larger sizes require more GPU memory during export and inference.
EXPORT_IMG_SIZE = 1920

# Precision: True for FP16, False for FP32
USE_FP16 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Validate Prerequisites ---
if DEVICE == "cpu":
    print("Error: TensorRT export requires an NVIDIA GPU and CUDA. CPU is not supported.")
    exit()

if not os.path.exists(PT_MODEL_PATH):
    print(f"Error: Model file not found at {PT_MODEL_PATH}")
    exit()

print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# --- Load Model ---
print(f"Loading YOLOv8 segmentation model from: {PT_MODEL_PATH}")
try:
    model = YOLO(PT_MODEL_PATH)
    model.to(DEVICE) # Move model to GPU before export
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Export to TensorRT ---
print("\nStarting TensorRT export...")
print(f"  Exporting for image size: {EXPORT_IMG_SIZE}x{EXPORT_IMG_SIZE}")
print(f"  Using FP16 precision: {USE_FP16}")

try:
    # The export function will create a .engine file in the same directory as the .pt file
    engine_path = model.export(
        format='engine',
        imgsz=EXPORT_IMG_SIZE,
        half=USE_FP16,
        device=DEVICE,
        verbose=True,       # Show detailed output during export
        workspace=4         # Workspace size in GB
    )
    print(f"TensorRT export successful")
    print(f"Engine file saved to: {engine_path}")

except ImportError as e:
     print(f"\nError during export: {e}")
     print("This often means TensorRT is not installed correctly or not found.")
     print("Please ensure you have installed NVIDIA TensorRT for your OS/CUDA version.")
     print("See: https://developer.nvidia.com/tensorrt")
except Exception as e:
    print(f"\nAn unexpected error occurred during export: {e}")
    print("This could be due to GPU memory limits, incompatible versions, or model issues.")
    print(f"Try reducing EXPORT_IMG_SIZE, increasing workspace, or checking logs.")