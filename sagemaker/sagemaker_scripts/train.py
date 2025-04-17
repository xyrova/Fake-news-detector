import os
import argparse
import pandas as pd
import torch
import numpy as np
import logging
import sys

from torch.nn.functional import softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def tokenize_function(examples, tokenizer):
    """Tokenizes text data."""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512) # Added max_length

def compute_metrics(p):
    """Computes accuracy for evaluation."""
    preds = p.predictions.argmax(-1)
    accuracy = accuracy_score(p.label_ids, preds)
    logger.info(f"Eval Accuracy: {accuracy}")
    return {"accuracy": accuracy}

# --- Main Training Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- SageMaker Environment/Hyperparameters ---
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased', help='Base model identifier from Hugging Face')
    parser.add_argument('--epochs', type=int, default=6, help='Number of training epochs')
    parser.add_argument('--train-batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--eval-batch-size', type=int, default=16, help='Evaluation batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for data splitting')

    # --- SageMaker Input/Output Paths ---
    # Default values are set by SageMaker environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model_output'))
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './results')) # For checkpoints etc.

    args = parser.parse_args()
    logger.info(f"Received arguments: {args}")

    # --- 1. Load and Preprocess Data ---
    logger.info("Loading and preprocessing data...")
    # Expecting train.csv directly in the train-dir channel
    train_data_path = os.path.join(args.train_dir, 'train.csv')
    logger.info(f"Attempting to load data from: {train_data_path}")
    if not os.path.exists(train_data_path):
         # Fallback for local testing if SM_CHANNEL_TRAIN isn't set correctly
         train_data_path_fallback = os.path.join(args.train_dir, 'balanced_news_dataset_1000.csv')
         logger.warning(f"train.csv not found at {train_data_path}. Trying {train_data_path_fallback}")
         train_data_path = train_data_path_fallback
         if not os.path.exists(train_data_path):
              logger.error(f"Training data CSV not found at {train_data_path} or fallback path.")
              sys.exit(1)


    data = pd.read_csv(train_data_path)
    logger.info(f"Data loaded successfully. Shape: {data.shape}")
    logger.info(f"Label distribution:\n{data['label'].value_counts()}")

    # Ensure required columns exist
    if 'title' not in data.columns or 'label' not in data.columns:
        logger.error("CSV must contain 'title' and 'label' columns.")
        sys.exit(1)

    # Drop rows with missing titles or labels if any
    data.dropna(subset=['title', 'label'], inplace=True)
    logger.info(f"Data shape after dropping NaNs: {data.shape}")

    # Ensure labels are integers
    data['label'] = data['label'].astype(int)

    # Stratified Split
    logger.info("Splitting data into train and test sets...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data['title'],
        data['label'],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=data['label']
    )
    logger.info(f"Train set size: {len(train_texts)}, Test set size: {len(test_texts)}")
    logger.info(f"Test set label distribution:\n{pd.Series(test_labels).value_counts()}")

    # --- 2. Tokenization ---
    logger.info(f"Loading tokenizer for model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create Hugging Face Datasets
    logger.info("Creating Hugging Face Datasets...")
    train_dataset_dict = {"text": train_texts.tolist(), "label": train_labels.tolist()}
    test_dataset_dict = {"text": test_texts.tolist(), "label": test_labels.tolist()}

    # Handle potential empty lists after split/dropna
    if not train_dataset_dict["text"] or not test_dataset_dict["text"]:
        logger.error("Train or test dataset is empty after processing. Check input data and split size.")
        sys.exit(1)

    train_dataset = Dataset.from_dict(train_dataset_dict)
    test_dataset = Dataset.from_dict(test_dataset_dict)


    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(lambda e: tokenize_function(e, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda e: tokenize_function(e, tokenizer), batched=True)
    logger.info("Tokenization complete.")

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # --- 3. Model Initialization ---
    logger.info(f"Loading model: {args.model_name}")
    num_labels = len(data['label'].unique())
    logger.info(f"Detected {num_labels} unique labels.")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # --- 4. Training Setup ---
    logger.info("Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir, # Checkpoints and logs go here
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",       # Save checkpoint at the end of each epoch
        logging_strategy="epoch",    # Log metrics at the end of each epoch
        # logging_steps=50,          # Alternative: Log every N steps
        # eval_steps=50,             # Alternative: Evaluate every N steps
        # save_steps=50,             # Alternative: Save every N steps
        load_best_model_at_end=True, # Load the best checkpoint at the end
        metric_for_best_model="accuracy", # Metric to determine the best model
        report_to="none", # Disable wandb/tensorboard reporting for simplicity here
        fp16=torch.cuda.is_available(), # Enable mixed precision if GPU available
    )

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset, # Use the test split for evaluation during training
        compute_metrics=compute_metrics,
        tokenizer=tokenizer # Pass tokenizer for consistent saving
    )

    # --- 5. Training ---
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")

    # --- 6. Evaluation (on the test set after training) ---
    logger.info("Evaluating final model on the test set...")
    results = trainer.evaluate(test_dataset)
    logger.info(f"Final Evaluation Results: {results}")
    logger.info(f"Final Test Accuracy: {results['eval_accuracy']:.4f}")

    # --- 7. Model Saving ---
    logger.info(f"Saving model and tokenizer to: {args.model_dir}")
    # SageMaker expects the model artifacts directly in SM_MODEL_DIR
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    logger.info("Model and tokenizer saved successfully.")

    # Optionally save evaluation results
    eval_results_path = os.path.join(args.output_dir, "final_eval_results.txt")
    with open(eval_results_path, "w") as f:
        f.write(f"Final Evaluation Results:\n{results}\n")
        f.write(f"Final Test Accuracy: {results['eval_accuracy']:.4f}\n")
    logger.info(f"Evaluation results saved to {eval_results_path}")