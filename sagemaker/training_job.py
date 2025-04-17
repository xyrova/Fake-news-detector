import sagemaker
from sagemaker.huggingface import HuggingFace
import boto3

sess = sagemaker.Session()
role = 'arn:aws:iam::529088299058:role/service-role/AmazonSageMaker-ExecutionRole-20250326T001497'


print(f"Using SageMaker Execution Role ARN: {role}")
# bucket = sess.default_bucket() # Or your specific bucket
bucket = 'sagemaker-bucket-pbl' # Your specified bucket

s3_train_data_uri = f's3://{bucket}/dataset/training/' # Points to the *directory*
s3_model_output_uri = f's3://{bucket}/models/fake-news-detector/'

# Define Hyperparameters (matching argparse defaults or overriding)
hyperparameters = {
    'model-name': 'distilbert-base-uncased',
    'epochs': 6,
    'train-batch-size': 16,
    'eval-batch-size': 16,
    'learning-rate': 2e-5,
    'weight-decay': 0.01,
    'test-size': 0.2,
    'random-state': 42
}

# Define the HuggingFace Estimator
huggingface_estimator = HuggingFace(
    entry_point          = 'train.py',          # Your training script
    source_dir           = './sagemaker_scripts', # Directory containing train.py and requirements.txt
    instance_type        = 'ml.g4dn.xlarge',    # Choose an appropriate instance type (GPU recommended)
    instance_count       = 1,
    role                 = role,
    transformers_version = '4.28',             # Specify versions compatible with your script/libs
    pytorch_version      = '2.0',               # Or match your torch version
    py_version           = 'py310',             # Or py38/py39 etc.
    hyperparameters      = hyperparameters,
    sagemaker_session    = sess,
    output_path          = s3_model_output_uri, # Where final model artifacts AND output data go
    # checkpoint_s3_uri  = f's3://{bucket}/checkpoints/fake-news/', # Optional: for spot training
    # distribution         = {'pytorchddp':{'enabled': True}} # For multi-GPU/multi-node
    # environment          = {"HUGGINGFACE_HUB_CACHE": "/tmp/.cache"} # Optional: cache location
)

# Launch the Training Job
huggingface_estimator.fit({'train': s3_train_data_uri}) # 'train' is the channel name

print(f"Training job name: {huggingface_estimator.latest_training_job.job_name}")
print(f"Model artifacts will be saved to: {huggingface_estimator.model_data}")
