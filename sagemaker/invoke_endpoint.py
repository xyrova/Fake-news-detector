import boto3
import json
import os
import argparse

# --- Configuration ---
# Replace with the actual name of your deployed endpoint
# You can find this in the SageMaker console or the output of deploy_endpoint.py
DEFAULT_ENDPOINT_NAME = "fake-news-rag-endpoint-1744910077" # <-- IMPORTANT: SET YOUR ENDPOINT NAME HERE

# Optionally configure AWS Region (often picked up from default config)
# AWS_REGION = "us-east-1" # Example region

def invoke_sagemaker_endpoint(endpoint_name, input_text, region_name=None):
    """
    Invokes the SageMaker endpoint with the provided text.

    Args:
        endpoint_name (str): The name of the deployed SageMaker endpoint.
        input_text (str): The news text to classify.
        region_name (str, optional): The AWS region name. Defaults to None (uses default).

    Returns:
        dict: The prediction result dictionary from the endpoint, or None if an error occurs.
    """
    print(f"Invoking endpoint: {endpoint_name}")
    print(f"Input Text (snippet): {input_text[:150]}...")

    # Create a SageMaker Runtime client
    # Explicitly pass region if needed, otherwise it uses default from config/env
    runtime_kwargs = {}
    if region_name:
        runtime_kwargs['region_name'] = region_name
    try:
        sagemaker_runtime = boto3.client('sagemaker-runtime', **runtime_kwargs)
    except Exception as e:
        print(f"Error creating SageMaker Runtime client: {e}")
        print("Ensure AWS credentials and region are configured correctly.")
        return None

    # Prepare the payload in the format expected by inference.py's input_fn
    payload = {"text": input_text}
    try:
        json_payload = json.dumps(payload)
    except TypeError as e:
        print(f"Error encoding payload to JSON: {e}")
        return None

    # Invoke the endpoint
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json', # Matches input_fn expectation
            Accept='application/json',      # Matches output_fn expectation
            Body=json_payload
        )
        print("Endpoint invoked successfully.")

        # Read the response body (it's a streaming body)
        response_body_bytes = response['Body'].read()
        response_body_str = response_body_bytes.decode('utf-8')

        # Parse the JSON response
        result = json.loads(response_body_str)
        return result

    except sagemaker_runtime.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        error_message = e.response.get("Error", {}).get("Message")
        print(f"SageMaker ClientError invoking endpoint: {error_code} - {error_message}")
        if "ModelError" in str(e):
             print("----> This often indicates an error within your inference.py script (model_fn, input_fn, predict_fn, or output_fn). Check the CloudWatch logs for the endpoint.")
        elif "ValidationError" in str(e):
             print("----> This often indicates an issue with the request format (e.g., incorrect ContentType, malformed Body).")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from endpoint: {e}")
        print(f"Raw response body: {response_body_str}") # Print raw response for debugging
        return None
    except Exception as e:
        print(f"An unexpected error occurred during endpoint invocation: {e}")
        return None


def map_prediction_to_label(prediction_code):
    """Maps the numerical prediction code to a human-readable label."""
    if prediction_code == 1:
        return "Real"
    elif prediction_code == 2:
        return "Fake"
    elif prediction_code == 3:
        return "Out of Context"
    elif prediction_code == -1:
        return "Error During Prediction"
    else:
        return f"Unknown ({prediction_code})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the SageMaker Fake News RAG endpoint.")
    parser.add_argument('--endpoint-name', type=str, default=os.environ.get('SAGEMAKER_ENDPOINT_NAME', DEFAULT_ENDPOINT_NAME),
                        help='The name of the deployed SageMaker endpoint.')
    parser.add_argument('--region', type=str, default=os.environ.get('AWS_REGION'),
                        help='Optional AWS region (if not using default config).')
    parser.add_argument('input_text', type=str, nargs='?', default="",
                        help='The news text to classify (optional, will prompt if not provided).')

    args = parser.parse_args()

    endpoint_to_test = args.endpoint_name
    text_to_classify = args.input_text

    # --- Verify Endpoint Name ---
    if not endpoint_to_test or "fake-news-rag-endpoint-XXXXXXXXXX" in endpoint_to_test:
        print("-" * 50)
        print("ERROR: Please set the 'DEFAULT_ENDPOINT_NAME' variable at the top")
        print("       of this script or provide it using --endpoint-name.")
        print("-" * 50)
        exit()

    # --- Get Input Text ---
    if not text_to_classify:
        print("\nPlease enter the news text you want to classify:")
        text_to_classify = input("> ")
        if not text_to_classify.strip():
            print("Input text cannot be empty.")
            exit()

    # --- Invoke Endpoint ---
    prediction_result = invoke_sagemaker_endpoint(endpoint_to_test, text_to_classify, args.region)

    # --- Display Results ---
    print("\n--- Results ---")
    if prediction_result:
        pred_code = prediction_result.get('prediction', 'N/A')
        explanation = prediction_result.get('explanation', 'N/A')
        error_msg = prediction_result.get('error')

        if error_msg:
            print(f"Prediction Error: {error_msg}")
        elif 'prediction' in prediction_result:
            label = map_prediction_to_label(pred_code)
            print(f"Prediction Code: {pred_code}")
            print(f"Prediction Label: {label}")
            print(f"\nExplanation:\n{explanation}")
        else:
            print("Received an unexpected response format:")
            print(prediction_result)
    else:
        print("Failed to get a prediction from the endpoint.")

    print("-------------")
