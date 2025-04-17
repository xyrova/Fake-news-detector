# inference_code/inference.py

import os
import json
import logging
import sys
import torch
import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings
from torch.nn.functional import softmax

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Ensure these match the artifacts in your model.tar.gz and training setup
# CLASSIFIER_MODEL_DIR is handled by model_fn's input 'model_dir'
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_FILE = "index.faiss"
DATA_MAP_FILE = "data_map.pkl"
SIMILARITY_THRESHOLD = 0.2 # Adjust based on validation if needed

# Global dictionary to hold loaded models/components (initialized in model_fn)
model_components = {}

def model_fn(model_dir):
    """
    Loads the classifier, tokenizer, sentence transformer, FAISS index, and data map.
    Runs once when the SageMaker endpoint container starts.
    """
    logger.info(f"Entering model_fn. Loading artifacts from: {model_dir}")
    global model_components

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Classifier Model and Tokenizer (from the bundled training output)
    try:
        logger.info(f"Loading classifier model from {model_dir}...")
        classifier_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        classifier_model.to(device)
        classifier_model.eval() # Set to evaluation mode
        logger.info("Classifier model loaded successfully.")

        logger.info(f"Loading classifier tokenizer from {model_dir}...")
        classifier_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logger.info("Classifier tokenizer loaded successfully.")

    except Exception as e:
        logger.error(f"Error loading classifier/tokenizer from {model_dir}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load classifier/tokenizer: {e}")

    # 2. Load Sentence Transformer Model (for RAG embeddings)
    try:
        logger.info(f"Loading Sentence Transformer model: {SENTENCE_MODEL_NAME}...")
        sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME, device=device)
        logger.info("Sentence Transformer model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Sentence Transformer model '{SENTENCE_MODEL_NAME}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load Sentence Transformer: {e}")

    # 3. Load FAISS Index (from the bundled artifacts)
    faiss_index_path = os.path.join(model_dir, FAISS_INDEX_FILE)
    try:
        logger.info(f"Loading FAISS index from: {faiss_index_path}...")
        if not os.path.exists(faiss_index_path):
             raise FileNotFoundError(f"FAISS index file '{FAISS_INDEX_FILE}' not found in {model_dir}")

        faiss_index = faiss.read_index(faiss_index_path)
        logger.info(f"FAISS index loaded successfully. Index size: {faiss_index.ntotal}")
    except Exception as e:
        logger.error(f"Error loading FAISS index from '{faiss_index_path}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load FAISS index: {e}")

    # 4. Load Data Map (from the bundled artifacts)
    data_map_path = os.path.join(model_dir, DATA_MAP_FILE)
    try:
        logger.info(f"Loading data map from: {data_map_path}...")
        if not os.path.exists(data_map_path):
             raise FileNotFoundError(f"Data map file '{DATA_MAP_FILE}' not found in {model_dir}")

        data_map_df = pd.read_pickle(data_map_path)
        # Ensure required columns exist
        if 'title' not in data_map_df.columns or 'label' not in data_map_df.columns:
            raise ValueError("Data map DataFrame must contain 'title' and 'label' columns.")
        logger.info(f"Data map loaded successfully. Shape: {data_map_df.shape}")
    except Exception as e:
        logger.error(f"Error loading data map from '{data_map_path}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load data map: {e}")

    # Store loaded components globally for predict_fn
    model_components = {
        "classifier_model": classifier_model,
        "classifier_tokenizer": classifier_tokenizer,
        "sentence_model": sentence_model,
        "faiss_index": faiss_index,
        "data_map_df": data_map_df,
        "device": device
    }
    logger.info("model_fn complete. All components loaded.")
    return model_components # Return components for predict_fn


def input_fn(request_body, request_content_type):
    """
    Parses the incoming request body. Expects JSON with a 'text' key.
    """
    logger.info(f"Received request with Content-Type: {request_content_type}")
    if request_content_type == 'application/json':
        try:
            data = json.loads(request_body)
            if 'text' not in data or not isinstance(data['text'], str):
                raise ValueError("Missing or invalid 'text' key (must be a string) in JSON input")
            if not data['text'].strip():
                 raise ValueError("'text' key cannot be empty")
            logger.info(f"Parsed input text: {data['text'][:100]}...") # Log snippet
            return data['text']
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError parsing request body: {e}", exc_info=True)
            raise ValueError(f"Invalid JSON format: {e}")
        except ValueError as e:
            logger.error(f"ValueError parsing request: {e}", exc_info=True)
            raise # Re-raise specific value errors
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON request body: {e}", exc_info=True)
            raise ValueError(f"Could not parse JSON request: {e}")
    else:
        error_msg = f"Unsupported content type: {request_content_type}. Must be application/json."
        logger.error(error_msg)
        raise ValueError(error_msg)


def predict_fn(input_data, loaded_model_components):
    """
    Performs RAG-enhanced classification using loaded components.
    """
    logger.info("Starting predict_fn...")
    text = input_data # Text string from input_fn
    global model_components # Access components loaded by model_fn

    if not model_components:
         logger.error("Model components not loaded. model_fn might have failed.")
         # You might want to return an error response or raise an exception
         # depending on SageMaker's error handling preference for predict_fn
         return {"error": "Model components not initialized", "prediction": -1, "explanation": "Internal server error."}


    # Retrieve components
    try:
        classifier_model = model_components["classifier_model"]
        classifier_tokenizer = model_components["classifier_tokenizer"]
        sentence_model = model_components["sentence_model"]
        faiss_index = model_components["faiss_index"]
        data_map_df = model_components["data_map_df"]
        device = model_components["device"]
    except KeyError as e:
        logger.error(f"Missing component in model_components dictionary: {e}", exc_info=True)
        return {"error": f"Missing component: {e}", "prediction": -1, "explanation": "Internal server error."}


    # --- RAG Similarity Check ---
    try:
        logger.info("Generating embedding for input text...")
        with torch.no_grad(): # Ensure no gradients are calculated
             input_embedding = sentence_model.encode([text], convert_to_tensor=True, device=device)
        input_embedding_np = normalize_embeddings(input_embedding).cpu().numpy()
        logger.info("Input embedding generated.")

        logger.info("Searching FAISS index...")
        distances, indices = faiss_index.search(input_embedding_np, k=1)
        similarity = distances[0][0] # FAISS IndexFlatIP returns inner product
        similar_db_index = indices[0][0]
        logger.info(f"FAISS search complete. Nearest index: {similar_db_index}, Similarity (IP): {similarity:.4f}")
    except Exception as e:
        logger.error(f"Error during embedding or FAISS search: {e}", exc_info=True)
        return {"error": "Failed during RAG similarity check", "prediction": -1, "explanation": "Error during similarity search."}


    # --- Classification Logic ---
    if similarity < SIMILARITY_THRESHOLD:
        logger.info(f"Similarity {similarity:.4f} < {SIMILARITY_THRESHOLD}. Classifying as Out of Context.")
        prediction = 3 # 3 for Out of Context
        explanation = "It's out of context as no sufficiently similar news was found in the database."
    else:
        logger.info(f"Similarity {similarity:.4f} >= {SIMILARITY_THRESHOLD}. Classifying...")
        try:
            # Use classification model
            inputs = classifier_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()} # Move inputs to device

            with torch.no_grad():
                logits = classifier_model(**inputs).logits
            probabilities = softmax(logits, dim=1)
            max_prob, predicted_class_tensor = torch.max(probabilities, dim=1)
            predicted_class = predicted_class_tensor.item() 
            confidence = max_prob.item()
            logger.info(f"Classifier prediction: class={predicted_class}, confidence={confidence:.4f}")

            # Map 0/1 to 1/2 and get label name
            if predicted_class == 1:
                prediction = 1 # 1 for Real
                label_name = "Real"
            else:
                prediction = 2 # 2 for Fake
                label_name = "Fake"

            # Get similar article for explanation
            try:
                # Assuming data_map_df's index aligns directly with FAISS internal indices
                similar_row = data_map_df.iloc[similar_db_index]
                similar_title = similar_row['title']
                similar_label = similar_row['label'] # Should be 0 or 1
                similar_label_name = 'real' if similar_label == 1 else 'fake'
                logger.info(f"Retrieved similar item: Title='{similar_title[:50]}...', Label={similar_label_name}")
            except IndexError:
                 logger.warning(f"Index {similar_db_index} out of bounds for data_map_df. Using generic explanation.")
                 similar_title = "[Could not retrieve title]"
                 similar_label_name = "[unknown]"
            except Exception as e:
                 logger.error(f"Error retrieving similar item details: {e}", exc_info=True)
                 similar_title = "[Error retrieving title]"
                 similar_label_name = "[error]"

            explanation = (f"Classified as {label_name} with confidence {confidence:.2f}. "
                           f"It is similar to '{similar_title}' which is known to be {similar_label_name} "
                           f"(similarity score: {similarity:.2f}).")
            logger.info(f"Generated explanation: {explanation}")

        except Exception as e:
            logger.error(f"Error during classification or explanation generation: {e}", exc_info=True)
            return {"error": "Failed during classification", "prediction": -1, "explanation": "Error during classification."}

    # --- Prepare Result ---
    result = {
        "prediction": prediction,        # Integer: 1 (Real), 2 (Fake), 3 (Out of Context)
        "explanation": explanation       # String
    }
    logger.info(f"predict_fn complete. Result: {result}")
    return result


def output_fn(prediction_output, accept_content_type):
    """
    Formats the prediction output (result from predict_fn) into JSON.
    """
    logger.info(f"Formatting prediction for Accept type: {accept_content_type}")
    if accept_content_type == 'application/json':
        try:
            response_body = json.dumps(prediction_output)
            # SageMaker expects the function to return the response body directly
            return response_body
        except TypeError as e:
            logger.error(f"TypeError serializing prediction to JSON: {e}. Prediction was: {prediction_output}", exc_info=True)
            # Return a JSON error message
            return json.dumps({"error": f"TypeError during JSON serialization: {e}"})
        except Exception as e:
            logger.error(f"Unexpected error serializing prediction to JSON: {e}", exc_info=True)
            return json.dumps({"error": f"Could not serialize prediction to JSON: {e}"})

    else:
        error_msg = f"Unsupported accept content type: {accept_content_type}. Must be application/json."
        logger.error(error_msg)
        # Raise error or return a JSON error message
        # Raising is often better as it signals a client/server negotiation issue
        raise ValueError(error_msg)
