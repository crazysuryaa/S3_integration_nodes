"""Node mappings for the new S3 Integration system

Defines the mapping between node classes and their display names for ComfyUI.
"""

from .s3_config_node import S3ConfigNode
from .s3_image_node import S3ImageNode
from .s3_video_node import S3VideoNode

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "S3ConfigNode": S3ConfigNode,
    "S3ImageNode": S3ImageNode,
    "S3VideoNode": S3VideoNode,
}

# Display name mappings for ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "S3ConfigNode": "S3 Configuration",
    "S3ImageNode": "S3 Image Operations",
    "S3VideoNode": "S3 Video Operations",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]