# S3 Integration for ComfyUI

Simple S3 integration for ComfyUI with unified image and video nodes.

## Installation

1. Copy the entire `S3_integration_nodes` folder to your ComfyUI `custom_nodes` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Configure S3 credentials (see below)
4. Restart ComfyUI

## S3 Configuration

### Method 1: Environment Variables
Set these environment variables in your system:

```bash
S3_REGION=us-east-1
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
S3_BUCKET_NAME=your_bucket_name
```

### Method 2: .env File (Recommended)
1. Copy `.env.template` to `.env` in the S3_integration_nodes folder
2. Edit `.env` and fill in your actual S3 credentials:

```bash
S3_REGION=us-east-1
S3_ACCESS_KEY=your_actual_access_key
S3_SECRET_KEY=your_actual_secret_key
S3_BUCKET_NAME=your_actual_bucket_name
```

### Test Your Configuration
Run the test script to verify your S3 setup:
```bash
python test_s3_config.py
```

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install boto3 python-dotenv opencv-python pillow
   ```

2. **Configure S3 credentials:**
   Create a `.env` file or set environment variables:
   ```env
   S3_REGION=us-east-1
   S3_ACCESS_KEY=your_access_key
   S3_SECRET_KEY=your_secret_key
   S3_BUCKET_NAME=your_bucket_name
   ```

3. **Use the nodes in ComfyUI**

## Nodes

### S3 Configuration
- Configure and test S3 connection
- Set up credentials directly in ComfyUI

### S3 Image Node
- **Load images** from S3
- **Save images** to S3
- Supports JPEG, PNG, WEBP formats

### S3 Video Node
- **Load videos** from S3
- **Extract frames** from videos
- **Save videos** to S3
- **Create videos** from frame sequences

## Basic Usage

1. Add "S3 Configuration" node and enter your credentials
2. Add "S3 Image Node" or "S3 Video Node"
3. Set the operation type (load/save)
4. Specify the S3 path for your file
5. Connect to other ComfyUI nodes as needed

## Example S3 Paths
- Images: `photos/image.jpg`
- Videos: `videos/clip.mp4`
- With directories: `input/photos/image.png`

## Troubleshooting

### Common Errors and Solutions

**"Failed to create S3 client: Invalid endpoint"**
- Check that S3_ENDPOINT_URL is either empty (for AWS) or a valid URL
- If using AWS, leave S3_ENDPOINT_URL unset or empty

**"'NoneType' object has no attribute 'Bucket'"**
- S3 client failed to initialize
- Verify all required environment variables are set: S3_REGION, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET_NAME
- Run `python test_s3_config.py` to diagnose the issue

**"'NoneType' object has no attribute 'read'"**
- File download from S3 failed
- Check that the file exists in S3 at the specified path
- Verify your AWS credentials have read permissions for the bucket

**General troubleshooting steps:**
1. Run the test script: `python test_s3_config.py`
2. Check your S3 credentials and region
3. Verify the S3 path format (s3://bucket/path)
4. Ensure your AWS user has S3 read/write permissions
5. Check ComfyUI console for detailed error messages

For detailed configuration options, see `MIGRATION_GUIDE.md`.