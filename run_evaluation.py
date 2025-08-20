import yaml
import argparse
import logging
import json
import os

from src.utils.logging_utils import setup_logging
from src.evaluation.evaluator_unsloth import ModelEvaluator

def main(config_path: str):
    """Main function to run the evaluation."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    run_name = os.path.basename(config['adapter_path']) + "_evaluation"
    setup_logging(run_name)
    logging.info(f"Starting evaluation run for model adapter: {config['adapter_path']}")
    logging.info(f"Configuration used:\n{json.dumps(config, indent=2)}")

    # Instantiate and run the evaluator
    evaluator = ModelEvaluator(config)
    evaluator.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on a fine-tuned model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the evaluation YAML configuration file."
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        exit(1)
        
    main(args.config)