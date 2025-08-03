from datasets import load_dataset
import logging

def format_prompt(example):
    """Formats a single example into the prompt structure the model expects."""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    # This structure is based on the Alpaca prompt format
    if input_text:
        return {"text": f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"}
    else:
        return {"text": f"### Instruction:\n{instruction}\n\n### Response:\n{output}"}

def load_and_prepare_dataset(config: dict):
    """Loads an Alpaca-style dataset and formats it for the SFTTrainer."""
    logger = logging.getLogger()
    dataset_path = config['dataset_path']
    logger.info(f"Loading dataset from: {dataset_path}")
    
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    formatted_dataset = dataset.map(format_prompt)
    
    logger.info(f"Dataset loaded and formatted successfully with {len(formatted_dataset)} examples.")
    return formatted_dataset
