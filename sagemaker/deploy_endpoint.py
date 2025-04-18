# deploy_endpoint.py
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
import time

# --- Configuration ---
# !! IMPORTANT: Replace these placeholders with your actual values !!
role_arn = 'arn:aws:iam::529088299058:role/service-role/AmazonSageMaker-ExecutionRole-20250326T001497' # Replace with your SageMaker Execution Role ARN
s3_model_uri = 's3://sagemaker-bucket-pbl/models/fake-news-detector/huggingface-pytorch-training-2025-04-17-13-05-33-034/output/final_model.tar.gz' # Replace with the S3 URI of your final_model.tar.gz
endpoint_name = f'fake-news-rag-endpoint-{int(time.time())}' # Creates a unique name
instance_type = 'ml.m5.xlarge' # GPU recommended for Sentence Transformer, but m5.large (CPU) might work for testing
# instance_type = 'ml.m5.large'  # Example CPU instance type

# --- SageMaker Session ---
try:
    sagemaker_session = sagemaker.Session()
    print(f"SageMaker session created successfully in region: {sagemaker_session.boto_region_name}")
except Exception as e:
    print(f"Error creating SageMaker session. Ensure AWS credentials are configured. Error: {e}")
    exit()

                
print(f"Using Role ARN: {role_arn}")
print(f"Using Model S3 URI: {s3_model_uri}")

# --- Define the HuggingFace Model for Inference ---
try:
    huggingface_model = HuggingFaceModel(
       model_data=s3_model_uri,          # Path to your final_model.tar.gz on S3
       role=role_arn,                    # Your IAM role ARN
       entry_point='inference.py',       # Inference script
       source_dir='./inference_code',    # Directory containing inference.py and requirements.txt
       transformers_version='4.28',      # Use versions compatible with your artifacts/script
       pytorch_version='2.0',
       py_version='py310',
       sagemaker_session=sagemaker_session,
       env={'TORCH_COMPILE': '0'}
    )
    print("HuggingFaceModel object created.")
except Exception as e:
    print(f"Error creating HuggingFaceModel object: {e}")
    exit()

# --- Deploy the Endpoint ---
print(f"Deploying endpoint '{endpoint_name}' with instance type '{instance_type}'...")
print("This may take several minutes...")
try:
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    print("Endpoint deployment initiated successfully.")
    print(f"Waiting for endpoint '{endpoint_name}' to be in service...")
    # predictor.wait() # This will block until the endpoint is ready (optional)
    print("-" * 30)
    print(f"Endpoint Name: {predictor.endpoint_name}")
    print("Endpoint is deploying. You can monitor its status in the SageMaker console.")
    print("Once 'InService', you can invoke it using the endpoint name.")
    print("-" * 30)

except Exception as e:
    print(f"Error deploying endpoint: {e}")
    # Consider adding cleanup logic here if needed, e.g., delete endpoint if partially created
