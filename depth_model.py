import cv2
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForDepthEstimation

MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

# Load processor + model
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME, trust_remote_code=True)

def get_depth(frame):
    # Convert BGR → RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Interpolate depth to original camera size
    predicted = torch.nn.functional.interpolate(
        outputs.predicted_depth.unsqueeze(1),
        size=img.shape[:2],
        mode="bilinear",
        align_corners=False
    )

    depth_map = predicted.squeeze().cpu().numpy()
    return depth_map
