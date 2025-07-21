import os
import tempfile
import torch
import numpy as np

from .s3_client import get_s3_client
from .logging_config import get_logger

logger = get_logger(__name__)


class S3VideoNode:
    """Unified S3 Video Node for loading and saving videos"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (["load_video", "load_frames", "save_video", "save_frames_as_video"], {"default": "load_video"}),
                "s3_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "path/to/video.mp4"
                })
            },
            "optional": {
                "video_path": ("STRING", {"tooltip": "Local path to video file (for save_video)"}),
                "frames": ("IMAGE",),  # Required for save_frames_as_video
                "use_input_dir": ("BOOLEAN", {"default": True}),
                "use_output_dir": ("BOOLEAN", {"default": True}),
                "keep_local_copy": ("BOOLEAN", {"default": False}),
                "delete_local": ("BOOLEAN", {"default": False}),
                # Frame extraction options
                "frame_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 10000}),
                # Video creation options
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1}),
                "codec": (["mp4v", "libx264", "libx265"], {"default": "libx264"}),
                "quality": ("INT", {"default": 23, "min": 0, "max": 51})
            }
        }
    
    CATEGORY = "S3Integration"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("frames", "local_path", "s3_path", "frame_count")
    FUNCTION = "process_video"
    OUTPUT_NODE = True
    
    def process_video(self, operation, s3_path, video_path=None, frames=None, use_input_dir=True, use_output_dir=True,
                     keep_local_copy=False, delete_local=False, frame_rate=0.0, max_frames=0, fps=24.0, 
                     codec="libx264", quality=23):
        """Process video based on operation type"""
        
        if not s3_path.strip():
            raise ValueError("S3 path cannot be empty")
        
        if operation == "load_video":
            return self._load_video(s3_path, use_input_dir, keep_local_copy)
        elif operation == "load_frames":
            return self._load_video_frames(s3_path, use_input_dir, frame_rate, max_frames)
        elif operation == "save_video":
            if not video_path:
                raise ValueError("Video path is required for save_video operation")
            return self._save_video(video_path, s3_path, use_output_dir, delete_local)
        elif operation == "save_frames_as_video":
            if frames is None:
                raise ValueError("Frames input is required for save_frames_as_video operation")
            return self._save_frames_as_video(frames, s3_path, use_output_dir, fps, codec, quality)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _load_video(self, s3_path, use_input_dir=True, keep_local_copy=False):
        """Load video from S3"""
        
        try:
            s3_instance = get_s3_client()
            if s3_instance is None:
                raise RuntimeError("Failed to initialize S3 client. Check your S3 configuration.")
            
            # Construct full S3 path
            if use_input_dir:
                input_dir = os.getenv("S3_INPUT_DIR", "input/")
                full_s3_path = os.path.join(input_dir, s3_path).replace("\\", "/")
            else:
                full_s3_path = s3_path
            
            # Create local filename from S3 path
            filename = os.path.basename(s3_path)
            
            if keep_local_copy:
                # Save to persistent location
                local_dir = os.path.join("temp", "videos")
                os.makedirs(local_dir, exist_ok=True)
                local_path = os.path.join(local_dir, filename)
            else:
                # Use temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
                local_path = temp_file.name
                temp_file.close()
            
            # Download video from S3
            logger.info(f"Downloading video from S3: {full_s3_path}")
            downloaded_path = s3_instance.download_file(s3_path=full_s3_path, local_path=local_path)
            
            if downloaded_path is None:
                raise RuntimeError(f"Failed to download video from S3: {full_s3_path}")
            
            logger.info(f"Successfully loaded video from S3: {full_s3_path} -> {downloaded_path}")
            # Return empty tensor for frames, local path, s3 path, and 0 frame count
            empty_tensor = torch.zeros((1, 1, 1, 3))
            return (empty_tensor, downloaded_path, full_s3_path, 0)
            
        except Exception as e:
            error_msg = f"Failed to load video from S3 path '{s3_path}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _load_video_frames(self, s3_path, use_input_dir=True, frame_rate=0.0, max_frames=0):
        """Load video from S3 and extract frames"""
        
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV (cv2) is required for video frame extraction. Install with: pip install opencv-python")
        
        try:
            s3_instance = get_s3_client()
            if s3_instance is None:
                raise RuntimeError("Failed to initialize S3 client. Check your S3 configuration.")
            
            # Construct full S3 path
            if use_input_dir:
                input_dir = os.getenv("S3_INPUT_DIR", "input/")
                full_s3_path = os.path.join(input_dir, s3_path).replace("\\", "/")
            else:
                full_s3_path = s3_path
            
            # Create temporary file for video
            filename = os.path.basename(s3_path)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
            local_path = temp_file.name
            temp_file.close()
            
            # Download video from S3
            logger.info(f"Downloading video from S3: {full_s3_path}")
            downloaded_path = s3_instance.download_file(s3_path=full_s3_path, local_path=local_path)
            
            if downloaded_path is None:
                raise RuntimeError(f"Failed to download video from S3: {full_s3_path}")
            
            # Extract frames using OpenCV
            cap = cv2.VideoCapture(downloaded_path)
            frames = []
            frame_count = 0
            
            # Calculate frame skip if frame_rate is specified
            if frame_rate > 0:
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                frame_skip = max(1, int(video_fps / frame_rate))
            else:
                frame_skip = 1
            
            current_frame = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if current_frame % frame_skip == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to tensor format expected by ComfyUI
                    frame_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
                    frames.append(frame_tensor)
                    frame_count += 1
                    
                    # Check max frames limit
                    if max_frames > 0 and frame_count >= max_frames:
                        break
                
                current_frame += 1
            
            cap.release()
            
            # Clean up temporary file
            try:
                os.remove(downloaded_path)
            except:
                pass
            
            if not frames:
                raise RuntimeError("No frames could be extracted from the video")
            
            # Stack frames into a single tensor
            frames_tensor = torch.stack(frames, dim=0)
            
            logger.info(f"Successfully extracted {frame_count} frames from video: {full_s3_path}")
            return (frames_tensor, "", full_s3_path, frame_count)
            
        except Exception as e:
            error_msg = f"Failed to load video frames from S3 path '{s3_path}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _save_video(self, video_path, s3_path, use_output_dir=True, delete_local=False):
        """Save video to S3"""
        
        if not os.path.exists(video_path):
            raise ValueError(f"Local video file does not exist: {video_path}")
        
        try:
            s3_instance = get_s3_client()
            if s3_instance is None:
                raise RuntimeError("Failed to initialize S3 client. Check your S3 configuration.")
            
            # Construct full S3 path
            if use_output_dir:
                output_dir = os.getenv("S3_OUTPUT_DIR", "output/")
                full_s3_path = os.path.join(output_dir, s3_path).replace("\\", "/")
            else:
                full_s3_path = s3_path
            
            # Upload video to S3
            logger.info(f"Uploading video to S3: {video_path} -> {full_s3_path}")
            uploaded_path = s3_instance.upload_file(video_path, full_s3_path)
            
            if uploaded_path is None:
                raise RuntimeError(f"Failed to upload video to S3: {full_s3_path}")
            
            # Delete local file if requested
            if delete_local:
                try:
                    os.remove(video_path)
                    logger.info(f"Deleted local video file: {video_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete local video file: {e}")
            
            logger.info(f"Successfully saved video to S3: {uploaded_path}")
            empty_tensor = torch.zeros((1, 1, 1, 3))
            return (empty_tensor, video_path if not delete_local else "", uploaded_path, 0)
            
        except Exception as e:
            error_msg = f"Failed to save video to S3 path '{s3_path}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _save_frames_as_video(self, frames, s3_path, use_output_dir=True, fps=24.0, codec="libx264", quality=23):
        """Save video frames to S3 as a video file"""
        
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV (cv2) is required for video creation. Install with: pip install opencv-python")
        
        try:
            s3_instance = get_s3_client()
            if s3_instance is None:
                raise RuntimeError("Failed to initialize S3 client. Check your S3 configuration.")
            
            # Construct full S3 path
            if use_output_dir:
                output_dir = os.getenv("S3_OUTPUT_DIR", "output/")
                full_s3_path = os.path.join(output_dir, s3_path).replace("\\", "/")
            else:
                full_s3_path = s3_path
            
            # Create temporary video file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video_path = temp_file.name
            temp_file.close()
            
            # Convert frames tensor to numpy arrays
            if len(frames.shape) == 4:  # Batch of frames
                frame_list = []
                for i in range(frames.shape[0]):
                    frame = frames[i].cpu().numpy()
                    frame = (frame * 255).astype(np.uint8)
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_list.append(frame_bgr)
            else:
                raise ValueError("Expected 4D tensor (batch, height, width, channels)")
            
            if not frame_list:
                raise ValueError("No frames to save")
            
            # Get frame dimensions
            height, width = frame_list[0].shape[:2]
            
            # Define codec and create VideoWriter
            if codec == "mp4v":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif codec == "libx264":
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            elif codec == "libx265":
                fourcc = cv2.VideoWriter_fourcc(*'hev1')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
            # Write frames to video
            for frame in frame_list:
                out.write(frame)
            
            out.release()
            
            # Upload video to S3
            logger.info(f"Uploading video to S3: {temp_video_path} -> {full_s3_path}")
            uploaded_path = s3_instance.upload_file(temp_video_path, full_s3_path)
            
            if uploaded_path is None:
                raise RuntimeError(f"Failed to upload video to S3: {full_s3_path}")
            
            # Clean up temporary file
            try:
                os.remove(temp_video_path)
            except:
                pass
            
            logger.info(f"Successfully saved video from {len(frame_list)} frames to S3: {uploaded_path}")
            return (frames, "", uploaded_path, len(frame_list))
            
        except Exception as e:
            error_msg = f"Failed to save video frames to S3 path '{s3_path}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)