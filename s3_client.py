"""S3 Client for ComfyUI S3 Integration

A robust S3 client with comprehensive error handling and logging.
"""

import os
import boto3
from typing import Optional, List, Tuple
from botocore.exceptions import NoCredentialsError, ClientError
from botocore.config import Config
from dotenv import load_dotenv

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class S3ClientError(Exception):
    """Custom exception for S3 client errors"""
    pass


class S3Client:
    """Enhanced S3 client with robust error handling and logging"""
    
    def __init__(self, region: str, access_key: str, secret_key: str, 
                 bucket_name: str, endpoint_url: Optional[str] = None):
        """Initialize S3 client with configuration
        
        Args:
            region: AWS region
            access_key: AWS access key ID
            secret_key: AWS secret access key
            bucket_name: S3 bucket name
            endpoint_url: Custom S3 endpoint URL (optional)
        """
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        
        # Initialize client
        self._client = None
        self._resource = None
        self._initialize_client()
        
        # Set up default directories
        self.input_dir = os.getenv("S3_INPUT_DIR", "input/")
        self.output_dir = os.getenv("S3_OUTPUT_DIR", "output/")
        
        # Ensure directories exist
        self._ensure_directories_exist()
    
    def _initialize_client(self) -> None:
        """Initialize S3 client and resource"""
        if not all([self.region, self.access_key, self.secret_key, self.bucket_name]):
            raise S3ClientError("Missing required S3 configuration. Check environment variables.")
        
        try:
            # Configure S3 addressing style
            addressing_style = os.getenv("S3_ADDRESSING_STYLE", "auto")
            if addressing_style not in ["auto", "virtual", "path"]:
                logger.warning(f"Invalid S3_ADDRESSING_STYLE: {addressing_style}, using 'auto'")
                addressing_style = "auto"
            
            config = Config(
                s3={'addressing_style': addressing_style},
                retries={'max_attempts': 3, 'mode': 'adaptive'}
            )
            
            # Create S3 resource
            self._resource = boto3.resource(
                service_name='s3',
                region_name=self.region,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                endpoint_url=self.endpoint_url,
                config=config
            )
            
            # Create S3 client for operations that need it
            self._client = boto3.client(
                service_name='s3',
                region_name=self.region,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                endpoint_url=self.endpoint_url,
                config=config
            )
            
            # Test connection
            self._test_connection()
            
            logger.info(f"S3 client initialized successfully for bucket: {self.bucket_name}")
            
        except Exception as e:
            error_msg = f"Failed to initialize S3 client: {str(e)}"
            logger.error(error_msg)
            raise S3ClientError(error_msg)
    
    def _test_connection(self) -> None:
        """Test S3 connection by checking bucket access"""
        try:
            self._client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise S3ClientError(f"Bucket '{self.bucket_name}' does not exist")
            elif error_code == '403':
                raise S3ClientError(f"Access denied to bucket '{self.bucket_name}'")
            else:
                raise S3ClientError(f"Failed to access bucket: {str(e)}")
    
    def _ensure_directories_exist(self) -> None:
        """Ensure input and output directories exist in S3"""
        for directory in [self.input_dir, self.output_dir]:
            if not self.folder_exists(directory):
                self.create_folder(directory)
                logger.info(f"Created S3 directory: {directory}")
    
    @property
    def is_connected(self) -> bool:
        """Check if S3 client is properly connected"""
        return self._client is not None and self._resource is not None
    
    def folder_exists(self, folder_path: str) -> bool:
        """Check if a folder exists in S3
        
        Args:
            folder_path: S3 folder path to check
            
        Returns:
            True if folder exists, False otherwise
        """
        if not self.is_connected:
            return False
        
        try:
            # Ensure folder path ends with /
            if not folder_path.endswith('/'):
                folder_path += '/'
            
            bucket = self._resource.Bucket(self.bucket_name)
            objects = list(bucket.objects.filter(Prefix=folder_path, MaxKeys=1))
            return len(objects) > 0
            
        except Exception as e:
            logger.error(f"Failed to check folder existence '{folder_path}': {str(e)}")
            return False
    
    def create_folder(self, folder_path: str) -> bool:
        """Create a folder in S3
        
        Args:
            folder_path: S3 folder path to create
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            return False
        
        try:
            # Ensure folder path ends with /
            if not folder_path.endswith('/'):
                folder_path += '/'
            
            bucket = self._resource.Bucket(self.bucket_name)
            bucket.put_object(Key=folder_path)
            logger.debug(f"Created S3 folder: {folder_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create folder '{folder_path}': {str(e)}")
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in S3 with given prefix
        
        Args:
            prefix: S3 path prefix to filter files
            
        Returns:
            List of file keys
        """
        if not self.is_connected:
            return []
        
        try:
            bucket = self._resource.Bucket(self.bucket_name)
            files = [obj.key for obj in bucket.objects.filter(Prefix=prefix)
                    if not obj.key.endswith('/')]
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files with prefix '{prefix}': {str(e)}")
            return []
    
    def download_file(self, s3_path: str, local_path: str) -> Optional[str]:
        """Download file from S3 to local path
        
        Args:
            s3_path: S3 object key
            local_path: Local file path to save to
            
        Returns:
            Local path if successful, None otherwise
        """
        if not self.is_connected:
            logger.error("S3 client not connected")
            return None
        
        try:
            # Ensure local directory exists
            local_dir = os.path.dirname(local_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir, exist_ok=True)
            
            bucket = self._resource.Bucket(self.bucket_name)
            bucket.download_file(s3_path, local_path)
            
            logger.debug(f"Downloaded: {s3_path} -> {local_path}")
            return local_path
            
        except NoCredentialsError:
            logger.error("AWS credentials not available or invalid")
            return None
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.error(f"File not found in S3: {s3_path}")
            else:
                logger.error(f"Failed to download '{s3_path}': {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading '{s3_path}': {str(e)}")
            return None
    
    def upload_file(self, local_path: str, s3_path: str) -> Optional[str]:
        """Upload file from local path to S3
        
        Args:
            local_path: Local file path to upload
            s3_path: S3 object key to upload to
            
        Returns:
            S3 path if successful, None otherwise
        """
        if not self.is_connected:
            logger.error("S3 client not connected")
            return None
        
        if not os.path.exists(local_path):
            logger.error(f"Local file does not exist: {local_path}")
            return None
        
        try:
            bucket = self._resource.Bucket(self.bucket_name)
            bucket.upload_file(local_path, s3_path)
            
            logger.debug(f"Uploaded: {local_path} -> {s3_path}")
            return s3_path
            
        except NoCredentialsError:
            logger.error("AWS credentials not available or invalid")
            return None
        except Exception as e:
            logger.error(f"Failed to upload '{local_path}' to '{s3_path}': {str(e)}")
            return None
    
    def generate_save_path(self, filename_prefix: str, width: int = 0, height: int = 0) -> Tuple[str, str, int]:
        """Generate a unique save path for files
        
        Args:
            filename_prefix: Base filename prefix
            width: Image width (for template substitution)
            height: Image height (for template substitution)
            
        Returns:
            Tuple of (full_s3_folder, filename, counter)
        """
        # Replace template variables
        filename_prefix = filename_prefix.replace("%width%", str(width))
        filename_prefix = filename_prefix.replace("%height%", str(height))
        
        # Split path components
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))
        
        # Construct full S3 folder path
        if subfolder:
            full_s3_folder = os.path.join(self.output_dir, subfolder).replace("\\", "/")
        else:
            full_s3_folder = self.output_dir
        
        # Ensure folder exists
        if not self.folder_exists(full_s3_folder):
            self.create_folder(full_s3_folder)
        
        # Find next available counter
        counter = self._get_next_counter(full_s3_folder, filename)
        
        return full_s3_folder, filename, counter
    
    def _get_next_counter(self, folder_path: str, filename: str) -> int:
        """Get the next available counter for a filename
        
        Args:
            folder_path: S3 folder path
            filename: Base filename
            
        Returns:
            Next available counter
        """
        try:
            files = self.list_files(folder_path)
            
            # Extract counters from existing files
            counters = []
            for file_key in files:
                file_basename = os.path.basename(file_key)
                if file_basename.startswith(filename + "_"):
                    try:
                        # Extract counter from filename_counter.ext format
                        counter_part = file_basename[len(filename) + 1:]
                        counter = int(counter_part.split('.')[0])
                        counters.append(counter)
                    except (ValueError, IndexError):
                        continue
            
            return max(counters) + 1 if counters else 1
            
        except Exception as e:
            logger.warning(f"Failed to determine counter for '{filename}': {str(e)}")
            return 1


# Global S3 client instance
_s3_client_instance: Optional[S3Client] = None


def get_s3_client() -> Optional[S3Client]:
    """Get or create S3 client instance
    
    Returns:
        S3Client instance if successful, None otherwise
    """
    global _s3_client_instance
    
    if _s3_client_instance is not None and _s3_client_instance.is_connected:
        return _s3_client_instance
    
    try:
        endpoint_url = os.getenv("S3_ENDPOINT_URL")
        # Convert empty string to None for default AWS endpoint
        if endpoint_url == "":
            endpoint_url = None
            
        _s3_client_instance = S3Client(
            region=os.getenv("S3_REGION"),
            access_key=os.getenv("S3_ACCESS_KEY"),
            secret_key=os.getenv("S3_SECRET_KEY"),
            bucket_name=os.getenv("S3_BUCKET_NAME"),
            endpoint_url=endpoint_url
        )
        return _s3_client_instance
        
    except Exception as e:
        logger.error(f"Failed to create S3 client: {str(e)}")
        return None


# Backward compatibility function
def get_s3_instance() -> Optional[S3Client]:
    """Backward compatibility wrapper for get_s3_client"""
    return get_s3_client()