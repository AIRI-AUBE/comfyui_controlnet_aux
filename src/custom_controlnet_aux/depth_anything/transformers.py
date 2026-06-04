"""
Modern DepthAnything implementation using HuggingFace transformers.
Replaces legacy torch.hub.load DINOv2 backbone with transformers pipeline.
"""

import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from custom_controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, resolve_local_hf_repo_path, validate_local_transformers_repo


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

class DepthAnythingDetector:
    """DepthAnything depth estimation using HuggingFace transformers."""
    
    def __init__(self, model_name="LiheYoung/depth-anything-large-hf"):
        """Initialize DepthAnything with specified model."""
        self.model_name = model_name
        try:
            local_model_path = resolve_local_hf_repo_path(model_name)
            validate_local_transformers_repo(
                local_model_path,
                required_files=["config.json", "preprocessor_config.json"],
                weight_files=["model.safetensors", "pytorch_model.bin"],
                repo_label=model_name,
            )
            self.processor = AutoImageProcessor.from_pretrained(local_model_path)
            self.model = AutoModelForDepthEstimation.from_pretrained(local_model_path)
        except Exception as e:
            raise FileNotFoundError(
                "DepthAnything is configured for repo-local-only use. "
                f"Transformers assets for '{model_name}' were not found under the repo ckpts layout. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
        self.device = "cpu"

    @classmethod  
    def from_pretrained(cls, pretrained_model_or_path=None, filename="depth_anything_vitl14.pth"):
        """Create DepthAnything from pretrained model, mapping legacy names to HuggingFace models."""
        
        # Map legacy checkpoint names to modern HuggingFace models
        model_mapping = {
            "depth_anything_vitl14.pth": "LiheYoung/depth-anything-large-hf",
            "depth_anything_vitb14.pth": "LiheYoung/depth-anything-base-hf", 
            "depth_anything_vits14.pth": "LiheYoung/depth-anything-small-hf"
        }
        
        model_name = model_mapping.get(filename, "LiheYoung/depth-anything-large-hf")
        return cls(model_name=model_name)
    
    def to(self, device):
        """Move model to specified device."""
        self.model = self.model.to(device)
        self.device = device
        return self
        
    def __call__(self, input_image, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        """Perform depth estimation on input image."""
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        if isinstance(input_image, np.ndarray):
            pil_image = Image.fromarray(input_image)
        else:
            pil_image = input_image

        with torch.no_grad():
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

            predicted_depth = getattr(outputs, "predicted_depth", None)
            if predicted_depth is None:
                raise RuntimeError("DepthAnything model did not return predicted_depth")

            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=pil_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_array = depth.detach().float().cpu().numpy()

            depth_min = float(depth_array.min())
            depth_max = float(depth_array.max())
            if depth_max > depth_min:
                depth_array = (depth_array - depth_min) / (depth_max - depth_min) * 255.0
            else:
                depth_array = np.zeros_like(depth_array)

            depth_image = depth_array.clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(depth_image))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map