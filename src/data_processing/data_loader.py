from datasets import load_dataset
import logging

def format_prompt(example):
    """Formats a single example into the prompt structure the model expects."""
    prompt = example['prompt']
    instruction = prompt.get('instruction', '')
    input_text = prompt.get('input', '')
    output = prompt.get('output', '')

    if input_text:
        return {"text": f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"}
    else:
        return {"text": f"### Instruction:\n{instruction}\n\n### Response:\n{output}"}

def format_prompt_batch(examples):
    """Formats a batch of examples into the prompt structure the model expects."""
    
    # Initialize a list to hold the formatted text for each example in the batch
    formatted_texts = []
    
    # The input 'examples' is a dictionary where each key maps to a list of values.
    # We iterate through the examples by index.
    num_examples = len(examples['prompt'])
    
    for i in range(num_examples):
        # Extract the data for the current example
        prompt = examples['prompt'][i]
        instruction = prompt.get('instruction', '')
        input_text = prompt.get('input', '')
        output = prompt.get('output', '')

        # Construct the formatted string for the current example
        if input_text:
            formatted_texts.append(
                f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            )
        else:
            formatted_texts.append(
                f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            )
            
    # The function must return a dictionary where the key is the new column name
    # and the value is the list of formatted strings.
    return {"text": formatted_texts}

def load_and_prepare_datasets(config: dict):
    """Loads train, validation, and test datasets and formats them."""
    logger = logging.getLogger()
    dataset_paths = config['dataset_paths']
    logger.info(f"Loading datasets from paths: {dataset_paths}")
    
    # Load all splits
    train_dataset = load_dataset("json", data_files=dataset_paths['train'], split="train")
    validation_dataset = load_dataset("json", data_files=dataset_paths['validation'], split="train")
    test_dataset = load_dataset("json", data_files=dataset_paths['test'], split="train")
    
    # Apply formatting
    train_dataset = train_dataset.map(format_prompt_batch, batched=True)
    validation_dataset = validation_dataset.map(format_prompt_batch, batched=True)
    # We don't format the test set yet, as we might want to evaluate it differently
    train_dataset = train_dataset.remove_columns(set(train_dataset.features) - {"text"})
    validation_dataset = validation_dataset.remove_columns(set(validation_dataset.features) - {"text"})
    logger.info(f"Train dataset loaded: {len(train_dataset)} examples.")
    logger.info(f"Validation dataset loaded: {len(validation_dataset)} examples.")
    logger.info(f"Test dataset loaded: {len(test_dataset)} examples.")
    
    return train_dataset, validation_dataset, test_dataset