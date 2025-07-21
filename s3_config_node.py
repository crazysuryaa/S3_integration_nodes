"""S3 Configuration Node for ComfyUI

Provides S3 configuration and connection testing functionality.
"""

import os
from typing import Dict, Any, Tuple

from .s3_client import S3Client, S3ClientError
from .logging_config import get_logger

logger = get_logger(__name__)


class S3ConfigNode:
    """S3 Configuration Node for setting up and testing S3 connections"""
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "region": ("STRING", {
                    "default": os.getenv("S3_REGION", ""),
                    "multiline": False,
                    "placeholder": "us-east-1"
                }),
                "access_key": ("STRING", {
                    "default": os.getenv("S3_ACCESS_KEY", ""),
                    "multiline": False,
                    "placeholder": "Your AWS Access Key ID"
                }),
                "secret_key": ("STRING", {
                    "default": os.getenv("S3_SECRET_KEY", ""),
                    "multiline": False,
                    "placeholder": "Your AWS Secret Access Key"
                }),
                "bucket_name": ("STRING", {
                    "default": os.getenv("S3_BUCKET_NAME", ""),
                    "multiline": False,
                    "placeholder": "your-bucket-name"
                })
            },
            "optional": {
                "endpoint_url": ("STRING", {
                    "default": os.getenv("S3_ENDPOINT_URL", ""),
                    "multiline": False,
                    "placeholder": "https://s3.amazonaws.com (leave empty for AWS)"
                }),
                "input_dir": ("STRING", {
                    "default": os.getenv("S3_INPUT_DIR", "input/"),
                    "multiline": False,
                    "placeholder": "input/"
                }),
                "output_dir": ("STRING", {
                    "default": os.getenv("S3_OUTPUT_DIR", "output/"),
                    "multiline": False,
                    "placeholder": "output/"
                }),
                "test_connection": ("BOOLEAN", {"default": True}),
                "create_directories": ("BOOLEAN", {"default": True})
            }
        }
    
    CATEGORY = "S3Integration"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("status", "message", "connected")
    FUNCTION = "configure_s3"
    OUTPUT_NODE = True
    
    def configure_s3(self, region: str, access_key: str, secret_key: str, bucket_name: str,
                    endpoint_url: str = "", input_dir: str = "input/", output_dir: str = "output/",
                    test_connection: bool = True, create_directories: bool = True) -> Tuple[str, str, bool]:
        """Configure S3 connection and test it
        
        Args:
            region: AWS region
            access_key: AWS access key ID
            secret_key: AWS secret access key
            bucket_name: S3 bucket name
            endpoint_url: Custom S3 endpoint URL
            input_dir: Input directory path in S3
            output_dir: Output directory path in S3
            test_connection: Whether to test the connection
            create_directories: Whether to create input/output directories
            
        Returns:
            Tuple of (status, message, connected)
        """
        
        # Validate required fields
        if not all([region.strip(), access_key.strip(), secret_key.strip(), bucket_name.strip()]):
            return ("error", "Missing required S3 configuration fields", False)
        
        try:
            # Update environment variables
            os.environ["S3_REGION"] = region.strip()
            os.environ["S3_ACCESS_KEY"] = access_key.strip()
            os.environ["S3_SECRET_KEY"] = secret_key.strip()
            os.environ["S3_BUCKET_NAME"] = bucket_name.strip()
            
            if endpoint_url.strip():
                os.environ["S3_ENDPOINT_URL"] = endpoint_url.strip()
            elif "S3_ENDPOINT_URL" in os.environ:
                del os.environ["S3_ENDPOINT_URL"]
            
            if input_dir.strip():
                os.environ["S3_INPUT_DIR"] = input_dir.strip()
            if output_dir.strip():
                os.environ["S3_OUTPUT_DIR"] = output_dir.strip()
            
            # Create S3 client
            s3_client = S3Client(
                region=region.strip(),
                access_key=access_key.strip(),
                secret_key=secret_key.strip(),
                bucket_name=bucket_name.strip(),
                endpoint_url=endpoint_url.strip() if endpoint_url.strip() else None
            )
            
            if not test_connection:
                logger.info("S3 configuration set (connection test skipped)")
                return ("success", "S3 configuration set successfully (connection not tested)", True)
            
            # Test connection
            if not s3_client.is_connected:
                return ("error", "Failed to establish S3 connection", False)
            
            # Create directories if requested
            if create_directories:
                success_count = 0
                total_dirs = 2
                
                if s3_client.create_folder(input_dir.strip()):
                    success_count += 1
                    logger.debug(f"Input directory ensured: {input_dir}")
                
                if s3_client.create_folder(output_dir.strip()):
                    success_count += 1
                    logger.debug(f"Output directory ensured: {output_dir}")
                
                if success_count < total_dirs:
                    logger.warning("Some directories could not be created")
            
            # Test basic operations
            test_results = self._run_connection_tests(s3_client)
            
            if test_results["success"]:
                message = f"S3 connection successful. Bucket: {bucket_name}"
                if test_results["details"]:
                    message += f" ({test_results['details']})"
                logger.info(message)
                return ("success", message, True)
            else:
                error_msg = f"S3 connection test failed: {test_results['error']}"
                logger.error(error_msg)
                return ("error", error_msg, False)
            
        except S3ClientError as e:
            error_msg = f"S3 configuration error: {str(e)}"
            logger.error(error_msg)
            return ("error", error_msg, False)
        
        except Exception as e:
            error_msg = f"Unexpected error during S3 configuration: {str(e)}"
            logger.error(error_msg)
            return ("error", error_msg, False)
    
    def _run_connection_tests(self, s3_client: S3Client) -> Dict[str, Any]:
        """Run basic connection tests
        
        Args:
            s3_client: S3 client instance
            
        Returns:
            Dictionary with test results
        """
        try:
            # Test 1: List files in input directory
            input_files = s3_client.list_files(s3_client.input_dir)
            
            # Test 2: Check if output directory exists
            output_exists = s3_client.folder_exists(s3_client.output_dir)
            
            details = []
            if isinstance(input_files, list):
                details.append(f"{len(input_files)} files in input dir")
            
            if output_exists:
                details.append("output dir accessible")
            
            return {
                "success": True,
                "details": ", ".join(details),
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "details": None,
                "error": str(e)
            }