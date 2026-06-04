"""
OneFormer implementation using HuggingFace transformers for PyTorch 2.7 compatibility.
Provides equivalent functionality to the original detectron2 implementation.
"""
from contextlib import suppress
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image

# Import utilities
from ..util import HWC3, common_input_validate, resize_image_with_pad, HF_MODEL_NAME, resolve_local_hf_repo_path, validate_local_transformers_repo


# Ensure transformers never attempts to contact huggingface.co.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _validate_local_oneformer_assets(local_model_path):
    validate_local_transformers_repo(
        local_model_path,
        required_files=[
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ],
        weight_files=["model.safetensors", "pytorch_model.bin"],
        repo_label=local_model_path,
    )


def _build_oneformer_class_info(metadata):
    if not isinstance(metadata, dict):
        return None

    class_names = metadata.get("class_names")
    thing_ids = metadata.get("thing_ids")
    if not isinstance(class_names, list) or not isinstance(thing_ids, list):
        return None

    try:
        thing_ids = {int(class_id) for class_id in thing_ids}
    except (TypeError, ValueError):
        return None

    class_info = {}
    for key, value in metadata.items():
        if key in {"class_names", "thing_ids"}:
            continue
        if not isinstance(key, str) or not key.isdigit() or not isinstance(value, str):
            continue

        class_id = int(key)
        class_info[key] = {
            "name": value,
            "isthing": class_id in thing_ids,
        }

    if not class_info or len(class_info) != len(class_names):
        return None

    return class_info


def _read_oneformer_preprocessor_config(local_model_path):
    config_path = Path(local_model_path) / "preprocessor_config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_oneformer_metadata_candidates(local_model_path, repo_path, class_info_file):
    local_model_dir = Path(local_model_path)
    yield local_model_dir / class_info_file, str(local_model_dir)

    if not repo_path:
        return

    repo_dir = Path(repo_path)
    if repo_dir.is_dir():
        yield repo_dir / class_info_file, str(repo_dir)

    with suppress(FileNotFoundError):
        resolved_repo_path = resolve_local_hf_repo_path(repo_path)
        resolved_repo_dir = Path(resolved_repo_path)
        yield resolved_repo_dir / class_info_file, resolved_repo_path


class _PreparedOneFormerProcessorConfig:
    def __init__(self, local_model_path):
        self.local_model_path = local_model_path
        self.temp_dir = None
        self.processor_kwargs = {"local_files_only": True}

    def __enter__(self):
        preprocessor_config = _read_oneformer_preprocessor_config(self.local_model_path)
        class_info_file = preprocessor_config.get("class_info_file")
        repo_path = preprocessor_config.get("repo_path")

        if not class_info_file:
            return self.processor_kwargs

        for metadata_path, resolved_repo_path in _iter_oneformer_metadata_candidates(
            self.local_model_path,
            repo_path,
            class_info_file,
        ):
            if metadata_path.is_file():
                self.processor_kwargs.update(
                    class_info_file=class_info_file,
                    repo_path=resolved_repo_path,
                )
                return self.processor_kwargs

        class_info = _build_oneformer_class_info(preprocessor_config.get("metadata"))
        if class_info is None:
            raise FileNotFoundError(
                "OneFormer preprocessor metadata is incomplete for offline use. "
                f"Could not find local class metadata file '{class_info_file}' for {self.local_model_path}, "
                "and the embedded preprocessor metadata could not reconstruct it."
            )

        self.temp_dir = tempfile.TemporaryDirectory(prefix="oneformer_metadata_")
        metadata_path = Path(self.temp_dir.name) / class_info_file
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(class_info, handle)

        self.processor_kwargs.update(
            class_info_file=class_info_file,
            repo_path=self.temp_dir.name,
        )
        return self.processor_kwargs

    def __exit__(self, exc_type, exc_value, traceback):
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def _prepare_oneformer_processor_kwargs(local_model_path):
    return _PreparedOneFormerProcessorConfig(local_model_path)


class OneformerSegmentor:
    """
    OneFormer segmentation using HuggingFace transformers implementation.
    
    Uses equivalent models that are PyTorch 2.7 compatible and actively maintained:
    - Same architecture (OneFormer with Swin-Large backbone)
    - Same training datasets (COCO panoptic / ADE20K)
    - Professional colorized visualization output
    """
    
    def __init__(self, model_name):
        """Initialize OneFormer with HuggingFace transformers implementation."""
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        
        self.model_name = model_name
        try:
            local_model_path = resolve_local_hf_repo_path(model_name)
            _validate_local_oneformer_assets(local_model_path)
            with _prepare_oneformer_processor_kwargs(local_model_path) as processor_kwargs:
                self.processor = OneFormerProcessor.from_pretrained(local_model_path, **processor_kwargs)
            self.model = OneFormerForUniversalSegmentation.from_pretrained(
                local_model_path,
                local_files_only=True,
            )
        except Exception as e:
            raise FileNotFoundError(
                "OneFormer is configured for repo-local-only use. "
                f"Transformers assets for '{model_name}' were not found under the repo ckpts layout. "
                "Place the full model repository under ckpts or provide a local directory path. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
        self.device = "cpu"

    @classmethod  
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="250_16_swin_l_oneformer_ade20k_160k.pth", config_path=None):
        """Create OneFormer model from pretrained weights."""
        model_mapping = {
            "250_16_swin_l_oneformer_ade20k_160k.pth": "shi-labs/oneformer_ade20k_swin_large",
            "150_16_swin_l_oneformer_coco_100ep.pth": "shi-labs/oneformer_coco_swin_large"
        }
        
        if filename in model_mapping:
            model_name = model_mapping[filename]
        elif "coco" in filename.lower():
            model_name = "shi-labs/oneformer_coco_swin_large"
        else:
            model_name = "shi-labs/oneformer_ade20k_swin_large"
        
        return cls(model_name)

    def to(self, device):
        """Move model to specified device."""
        self.model = self.model.to(device) 
        self.device = device
        return self
        
    def __call__(self, input_image=None, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        """Process image for semantic segmentation."""
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        # Convert to PIL for processing
        if isinstance(input_image, np.ndarray):
            pil_image = Image.fromarray(input_image)
        else:
            pil_image = input_image
            
        # Process with HuggingFace pipeline
        semantic_inputs = self.processor(
            images=pil_image, 
            task_inputs=["semantic"], 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**semantic_inputs)
        
        # Post-process results
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil_image.size[::-1]]
        )[0]
        
        # Convert to colormap using professional color scheme
        seg_map = predicted_semantic_map.cpu().numpy().astype(np.uint8)
        detected_map = self._generate_professional_colormap(seg_map)
        detected_map = remove_pad(HWC3(detected_map))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
    
    def _generate_professional_colormap(self, seg_map):
        """Generate professional colormap for segmentation visualization."""
        height, width = seg_map.shape
        color_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        max_possible_classes = 200
        colors = self._generate_detectron2_style_palette(max_possible_classes)
        
        unique_classes = np.unique(seg_map)
        for class_id in unique_classes:
            mask = seg_map == class_id
            color_map[mask] = colors[class_id % len(colors)]
            
        return color_map
    
    def _generate_detectron2_style_palette(self, num_classes):
        """Generate professional color palette with good visual separation."""
        colors = np.zeros((num_classes, 3), dtype=np.uint8)
        
        colors[0] = [0, 0, 0]  # Background is black
        
        for i in range(1, num_classes):
            hue = (i * 137.508) % 360  # Golden angle for good distribution
            saturation = 0.6 + 0.4 * ((i % 3) / 2)  # 60-100% saturation
            value = 0.7 + 0.3 * ((i % 2))  # 70-100% value
            
            color_hsv = np.array([[[hue / 2, saturation * 255, value * 255]]], dtype=np.uint8)
            color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]
            colors[i] = color_rgb
            
        return colors


