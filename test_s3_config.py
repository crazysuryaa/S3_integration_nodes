#!/usr/bin/env python3
"""
Test script to verify S3 configuration and client initialization
"""

import os
import sys

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, using environment variables only")

# Import our modules
try:
    from s3_client import get_s3_client
    from logging_config import get_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the S3_integration_nodes directory")
    sys.exit(1)

# Setup logging
logger = get_logger("S3Test")

def test_s3_config():
    """Test S3 configuration and connection"""
    print("Testing S3 Configuration...")
    
    # Check environment variables
    required_vars = ["S3_REGION", "S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_BUCKET_NAME"]
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            print(f"✓ {var}: {'*' * min(len(value), 10)}")
    
    if missing_vars:
        print(f"✗ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    # Check optional variables
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    if endpoint_url:
        print(f"✓ S3_ENDPOINT_URL: {endpoint_url}")
    else:
        print("✓ S3_ENDPOINT_URL: (using default AWS endpoint)")
    
    # Test S3 client initialization
    print("\nTesting S3 client initialization...")
    try:
        s3_client = get_s3_client()
        if s3_client is None:
            print("✗ Failed to create S3 client")
            return False
        
        if s3_client.is_connected:
            print("✓ S3 client connected successfully")
            
            # Test basic operations
            print("\nTesting basic S3 operations...")
            
            # Test folder existence check
            input_dir = s3_client.input_dir
            output_dir = s3_client.output_dir
            
            print(f"✓ Input directory: {input_dir}")
            print(f"✓ Output directory: {output_dir}")
            
            # Test listing files
            try:
                files = s3_client.list_files(input_dir)
                print(f"✓ Found {len(files)} files in input directory")
            except Exception as e:
                print(f"⚠ Warning: Could not list files in input directory: {e}")
            
            return True
        else:
            print("✗ S3 client not connected")
            return False
            
    except Exception as e:
        print(f"✗ Error creating S3 client: {e}")
        return False

if __name__ == "__main__":
    success = test_s3_config()
    if success:
        print("\n🎉 S3 configuration test passed!")
    else:
        print("\n❌ S3 configuration test failed!")
        print("\nPlease check your environment variables and S3 credentials.")