import os
import tempfile
from PIL import Image, ImageOps
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


class S3ImageNode:
    """Unified S3 Image Node for loading and saving images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (["load", "save"], {"default": "load"}),
                "s3_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "path/to/image.jpg"
                })
            },
            "optional": {
                "image": ("IMAGE",),  # Required for save operation
                "use_input_dir": ("BOOLEAN", {"default": True}),
                "use_output_dir": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "format": (["JPEG", "PNG", "WEBP"], {"default": "JPEG"}),
                "keep_local_copy": ("BOOLEAN", {"default": False}),
                "delete_local": ("BOOLEAN", {"default": False}),
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
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "local_path", "s3_path")
    FUNCTION = "process_image"
    OUTPUT_NODE = True
    
    def process_image(self, operation, s3_path, image=None, use_input_dir=True, use_output_dir=True, 
                     filename_prefix="ComfyUI", quality=95, format="JPEG", keep_local_copy=False, delete_local=False,
                     use_workflow_config=False, region="", access_key="", secret_key="", bucket_name="", 
                     endpoint_url="", input_dir="input/", output_dir="output/"):
        """Process image based on operation type"""
        
        if not s3_path.strip():
            raise ValueError("S3 path cannot be empty")
        
        # Get S3 client with appropriate configuration
        s3_client = self._get_s3_client_with_config(
            use_workflow_config, region, access_key, secret_key,
            bucket_name, endpoint_url, input_dir, output_dir
        )
        
        if s3_client is None:
            raise RuntimeError("Failed to initialize S3 client")
        
        if operation == "load":
            return self._load_image(s3_path, use_input_dir, keep_local_copy, s3_client)
        elif operation == "save":
            if image is None:
                raise ValueError("Image input is required for save operation")
            return self._save_image(image, s3_path, use_output_dir, filename_prefix, quality, format, delete_local, s3_client)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _load_image(self, s3_path, use_input_dir=True, keep_local_copy=False, s3_instance=None):
        """Load image from S3"""
        
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
                local_dir = os.path.join("temp", "images")
                os.makedirs(local_dir, exist_ok=True)
                local_path = os.path.join(local_dir, filename)
            else:
                # Use temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
                local_path = temp_file.name
                temp_file.close()
            
            # Download image from S3
            logger.info(f"Downloading image from S3: {full_s3_path}")
            downloaded_path = s3_instance.download_file(full_s3_path, local_path)
            
            if downloaded_path is None:
                raise RuntimeError(f"Failed to download image from S3: {full_s3_path}")
            
            # Load and process image
            img = Image.open(downloaded_path)
            img = ImageOps.exif_transpose(img)  # Handle EXIF rotation
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to tensor
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)[None,]
            
            # Clean up temporary file if not keeping local copy
            if not keep_local_copy:
                try:
                    os.remove(downloaded_path)
                except:
                    pass
            
            logger.info(f"Successfully loaded image from S3: {full_s3_path}")
            return (img_tensor, downloaded_path if keep_local_copy else "", full_s3_path)
            
        except Exception as e:
            error_msg = f"Failed to load image from S3 path '{s3_path}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _save_image(self, image, s3_path, use_output_dir=True, filename_prefix="ComfyUI", 
                   quality=95, format="JPEG", delete_local=False, s3_instance=None):
        """Save image to S3"""
        
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
                if use_output_dir:
                    output_dir = getattr(s3_instance, 'output_dir', os.getenv("S3_OUTPUT_DIR", "output/"))
                    full_s3_path = os.path.join(output_dir, s3_path).replace("\\", "/")
                else:
                    full_s3_path = s3_path
            
            # Convert tensor to PIL Image
            i = 255. * image[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Create temporary file
            file_extension = format.lower() if format.lower() != 'jpeg' else 'jpg'
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
            temp_path = temp_file.name
            temp_file.close()
            
            # Save image with specified format and quality
            save_kwargs = {}
            if format in ['JPEG', 'WEBP']:
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            
            img.save(temp_path, format=format, **save_kwargs)
            
            # Upload to S3
            logger.info(f"Uploading image to S3: {temp_path} -> {full_s3_path}")
            uploaded_path = s3_instance.upload_file(temp_path, full_s3_path)
            
            if uploaded_path is None:
                raise RuntimeError(f"Failed to upload image to S3: {full_s3_path}")
            
            # Clean up or keep local file
            if delete_local:
                try:
                    os.remove(temp_path)
                    logger.info(f"Deleted local image file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete local image file: {e}")
            
            logger.info(f"Successfully saved image to S3: {uploaded_path}")
            return (image, temp_path if not delete_local else "", uploaded_path)
            
        except Exception as e:
            error_msg = f"Failed to save image to S3 path '{s3_path}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _get_s3_client_with_config(self, use_workflow_config, region, access_key, secret_key, 
                                  bucket_name, endpoint_url, input_dir, output_dir):
        """Get S3 client with either workflow config or environment variables"""
        
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