import os
import tempfile
import torch
import numpy as np

# Handle both relative and absolute imports
try:
    from .s3_client import get_s3_client
    from .logging_config import get_logger
except ImportError:
    from s3_client import get_s3_client
    from logging_config import get_logger

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
                "quality": ("INT", {"default": 23, "min": 0, "max": 51}),
                # S3 Configuration
                "use_workflow_config": ("BOOLEAN", {"default": False}),
                "region": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "us-east-1"
                }),
                "access_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Your AWS Access Key ID"
                }),
                "secret_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Your AWS Secret Access Key"
                }),
                "bucket_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "your-bucket-name"
                }),
                "endpoint_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "https://s3.amazonaws.com (leave empty for AWS)"
                }),
                "input_dir": ("STRING", {
                    "default": "input/",
                    "multiline": False,
                    "placeholder": "input/"
                }),
                "output_dir": ("STRING", {
                    "default": "output/",
                    "multiline": False,
                    "placeholder": "output/"
                })
            }
        }
    
    CATEGORY = "S3Integration"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("frames", "local_path", "s3_path", "frame_count")
    FUNCTION = "process_video"
    OUTPUT_NODE = True
    
    def process_video(self, operation, s3_path, video_path=None, frames=None, use_input_dir=True, use_output_dir=True,
                     keep_local_copy=False, delete_local=False, frame_rate=0.0, max_frames=0, fps=24.0, 
                     codec="libx264", quality=23, use_workflow_config=False, region="", access_key="",
                     secret_key="", bucket_name="", endpoint_url="", input_dir="input/", output_dir="output/"):
        """Process video based on operation type"""
        
        if not s3_path.strip():
            raise ValueError("S3 path cannot be empty")
        
        # Get S3 client with appropriate configuration
        s3_client = self._get_s3_client_with_config(
            use_workflow_config, region, access_key, secret_key,
            bucket_name, endpoint_url, input_dir, output_dir
        )
        
        if s3_client is None:
            raise RuntimeError("Failed to initialize S3 client")
        
        if operation == "load_video":
            return self._load_video(s3_path, use_input_dir, keep_local_copy, s3_client)
        elif operation == "load_frames":
            return self._load_video_frames(s3_path, use_input_dir, frame_rate, max_frames, s3_client)
        elif operation == "save_video":
            if not video_path:
                raise ValueError("Video path is required for save_video operation")
            return self._save_video(video_path, s3_path, s3_client, use_output_dir, delete_local)
        elif operation == "save_frames_as_video":
            if frames is None:
                raise ValueError("Frames input is required for save_frames_as_video operation")
            return self._save_frames_as_video(frames, s3_path, s3_client, use_output_dir, fps, codec, quality)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _load_video(self, s3_path, use_input_dir=True, keep_local_copy=False, s3_instance=None):
        """Load video from S3"""
        
        try:
            if s3_instance is None:
                s3_instance = get_s3_client()
                if s3_instance is None:
                    raise RuntimeError("Failed to initialize S3 client. Check your S3 configuration.")
            
            # Parse S3 path - handle both full S3 URLs and relative paths
            if s3_path.startswith('s3://'):
                # Full S3 URL: extract the key part after bucket name
                parts = s3_path.replace('s3://', '').split('/', 1)
                if len(parts) > 1:
                    full_s3_path = parts[1]  # Use key without bucket name
                else:
                    raise ValueError(f"Invalid S3 URL format: {s3_path}")
            else:
                # Relative path - apply directory prefix if requested
                if use_input_dir:
                    input_dir = getattr(s3_instance, 'input_dir', os.getenv("S3_INPUT_DIR", "input/"))
                    full_s3_path = os.path.join(input_dir, s3_path).replace("\\", "/")
                else:
                    full_s3_path = s3_path
            
            # Create local filename from S3 path
            if s3_path.startswith('s3://'):
                # Extract filename from S3 URL
                filename = os.path.basename(full_s3_path)
            else:
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
            downloaded_path = s3_instance.download_file(full_s3_path, local_path)
            
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
    
    def _load_video_frames(self, s3_path, use_input_dir=True, frame_rate=0.0, max_frames=0, s3_instance=None):
        """Load video from S3 and extract frames"""
        
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV (cv2) is required for video frame extraction. Install with: pip install opencv-python")
        
        try:
            if s3_instance is None:
                s3_instance = get_s3_client()
                if s3_instance is None:
                    raise RuntimeError("Failed to initialize S3 client. Check your S3 configuration.")
            
            # Parse S3 path - handle both full S3 URLs and relative paths
            if s3_path.startswith('s3://'):
                # Full S3 URL: extract the key part after bucket name
                parts = s3_path.replace('s3://', '').split('/', 1)
                if len(parts) > 1:
                    full_s3_path = parts[1]  # Use key without bucket name
                else:
                    raise ValueError(f"Invalid S3 URL format: {s3_path}")
            else:
                # Relative path - apply directory prefix if requested
                if use_input_dir:
                    input_dir = getattr(s3_instance, 'input_dir', os.getenv("S3_INPUT_DIR", "input/"))
                    full_s3_path = os.path.join(input_dir, s3_path).replace("\\", "/")
                else:
                    full_s3_path = s3_path
            
            # Create temporary file for video
            if s3_path.startswith('s3://'):
                # Extract filename from S3 URL
                filename = os.path.basename(full_s3_path)
            else:
                filename = os.path.basename(s3_path)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
            local_path = temp_file.name
            temp_file.close()
            
            # Download video from S3
            logger.info(f"Downloading video from S3: {full_s3_path}")
            downloaded_path = s3_instance.download_file(full_s3_path, local_path)
            
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
    
    def _save_video(self, video_path, s3_path, s3_client, use_output_dir=True, delete_local=False):
        """Save video to S3"""
        
        if not os.path.exists(video_path):
            raise ValueError(f"Local video file does not exist: {video_path}")
        
        try:
            s3_instance = s3_client
            if s3_instance is None:
                raise RuntimeError("Failed to initialize S3 client. Check your S3 configuration.")
            
            # Parse S3 path - handle both full S3 URLs and relative paths
            if s3_path.startswith('s3://'):
                # Full S3 URL: extract the key part after bucket name
                parts = s3_path.replace('s3://', '').split('/', 1)
                if len(parts) > 1:
                    full_s3_path = parts[1]  # Use key without bucket name
                else:
                    raise ValueError(f"Invalid S3 URL format: {s3_path}")
            else:
                # Relative path - apply directory prefix if requested
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
    
    def _save_frames_as_video(self, frames, s3_path, s3_client, use_output_dir=True, fps=24.0, codec="libx264", quality=23):
        """Save video frames to S3 as a video file"""
        
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV (cv2) is required for video creation. Install with: pip install opencv-python")
        
        try:
            s3_instance = s3_client
            if s3_instance is None:
                raise RuntimeError("Failed to initialize S3 client. Check your S3 configuration.")
            
            # Parse S3 path - handle both full S3 URLs and relative paths
            if s3_path.startswith('s3://'):
                # Full S3 URL: extract the key part after bucket name
                parts = s3_path.replace('s3://', '').split('/', 1)
                if len(parts) > 1:
                    full_s3_path = parts[1]  # Use key without bucket name
                else:
                    raise ValueError(f"Invalid S3 URL format: {s3_path}")
            else:
                # Relative path - apply directory prefix if requested
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
            
            # Define codec and create VideoWriter with fallback support
            codec_options = [
                (codec, None),  # Try requested codec first
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v'))  # Fallback to mp4v
            ]
            
            # Map codec names to fourcc codes
            if codec == "mp4v":
                primary_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif codec == "libx264":
                primary_fourcc = cv2.VideoWriter_fourcc(*'X264')  # Use X264 for better compatibility
            elif codec == "libx265":
                primary_fourcc = cv2.VideoWriter_fourcc(*'X265')  # Use X265 for H.265
            else:
                primary_fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default fallback
            
            codec_options[0] = (codec, primary_fourcc)
            
            out = None
            used_codec = None
            
            # Try codecs in order until one works
            for codec_name, fourcc in codec_options:
                try:
                    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                    if out.isOpened():
                        used_codec = codec_name
                        logger.info(f"Successfully initialized VideoWriter with codec: {codec_name}")
                        break
                    else:
                        out.release()
                        out = None
                except Exception as e:
                    logger.warning(f"Failed to initialize VideoWriter with codec {codec_name}: {e}")
                    if out:
                        out.release()
                        out = None
            
            if out is None or not out.isOpened():
                raise RuntimeError(f"Failed to initialize VideoWriter with any codec. Tried: {[c[0] for c in codec_options]}")
            
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
    
    def _get_s3_client_with_config(self, use_workflow_config, region=None, access_key=None, secret_key=None, 
                                   bucket_name=None, endpoint_url=None, input_dir=None, output_dir=None):
        """Get S3 client using either workflow config or environment variables"""
        
        try:
            if use_workflow_config:
                # Import S3Client class
                from .s3_client import S3Client
                
                # Validate required workflow config
                if not all([region.strip(), access_key.strip(), secret_key.strip(), bucket_name.strip()]):
                    raise ValueError("Missing required S3 configuration in workflow parameters")
                
                # Create S3 client with workflow config
                s3_client = S3Client(
                    region=region.strip(),
                    access_key=access_key.strip(),
                    secret_key=secret_key.strip(),
                    bucket_name=bucket_name.strip(),
                    endpoint_url=endpoint_url.strip() if endpoint_url.strip() else None
                )
                
                # Override input/output directories if provided
                if input_dir.strip():
                    s3_client.input_dir = input_dir.strip()
                if output_dir.strip():
                    s3_client.output_dir = output_dir.strip()
                
                logger.info("Using S3 configuration from workflow parameters")
                return s3_client
            else:
                # Use environment variables (existing behavior)
                s3_client = get_s3_client()
                if s3_client is None:
                    raise ValueError("Failed to get S3 client from environment variables")
                
                logger.info("Using S3 configuration from environment variables")
                return s3_client
                
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            return None