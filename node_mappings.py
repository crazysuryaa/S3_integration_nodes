"""Node mappings for the new S3 Integration system

Defines the mapping between node classes and their display names for ComfyUI.
"""

# Handle both relative and absolute imports
try:
    from .s3_config_node import S3ConfigNode
    from .s3_image_node import S3ImageNode
    from .s3_video_node import S3VideoNode
    from .s3_audio_node import S3AudioNode
except ImportError:
    from s3_config_node import S3ConfigNode
    from s3_image_node import S3ImageNode
    from s3_video_node import S3VideoNode
    from s3_audio_node import S3AudioNode

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "S3ConfigNode": S3ConfigNode,
    "S3ImageNode": S3ImageNode,
    "S3VideoNode": S3VideoNode,
    "S3AudioNode": S3AudioNode,
}

# Display name mappings for ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "S3ConfigNode": "S3 Configuration",
    "S3ImageNode": "S3 Image Operations",
    "S3VideoNode": "S3 Video Operations",
    "S3AudioNode": "S3 Audio Operations",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]