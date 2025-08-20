import yaml
import argparse
import json
import os

from src.utils.logging_utils import setup_logging
from src.data_processing.data_loader import load_and_prepare_datasets
from src.model_training.trainer_unsloth import ModelTrainerUnsloth

def main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger = setup_logging(config['run_name'])
    logger.info("Starting new fine-tuning run.")
    
    logger.info("Configuration used:")
    logger.info(json.dumps(config, indent=2))

    # --- Data Loading and Preparation ---
    train_ds, val_ds, test_ds = load_and_prepare_datasets(config)

    # --- Model Training and Validation ---
    trainer = ModelTrainerUnsloth(config)
    trainer.train(train_ds, val_ds)
    logger.info("Fine-tuning run finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a fine-tuning experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        exit(1)
        
    main(args.config)