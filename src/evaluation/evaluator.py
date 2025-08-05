import logging
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

class ModelEvaluator:
    """Encapsulates the logic for evaluating a fine-tuned model."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger()
        self.model = None
        self.tokenizer = None

    def _load_model_and_tokenizer(self):
        """
        Loads the base model and applies the fine-tuned LoRA adapters.
        """
        base_model_path = self.config['base_model_path']
        adapter_path = self.config['adapter_path']
        
        self.logger.info(f"Loading base model from: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_folder="offload"
        )

        self.logger.info(f"Loading tokenizer from: {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info(f"Loading PEFT adapter from: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path, device_map="auto", offload_folder="offload")
        
        # For faster inference, you can merge the adapter layers into the base model.
        # This requires more memory but is faster.
        # self.model = self.model.merge_and_unload()
        
        self.model.eval()

    def _load_dataset(self):
        """Loads the test dataset."""
        path = self.config['test_dataset_path']
        self.logger.info(f"Loading test dataset from: {path}")
        return load_dataset("json", data_files=path, split="train")

    def run(self):
        """Executes the full evaluation pipeline."""
        self._load_model_and_tokenizer()
        test_dataset = self._load_dataset()

        prompt_template = self.config['prompt_template']
        generation_params = self.config.get('generation_params', {})
        results = []

        self.logger.info("Generating predictions on the test set...")
        for example in tqdm(test_dataset):
            prompt_text = example['prompt']
            prompt = prompt_template.format(
                instruction=prompt_text['instruction'],
                input=prompt_text['input']
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_params)
            
            prediction_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_only = prediction_text.split("### Response:")[1].strip()
            ground_truth = prompt_text['output']
            results.append({
                "input_prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": response_only
            })

        output_path = self.config['predictions_output_path']
        self.logger.info(f"Saving {len(results)} predictions to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info("Evaluation run finished successfully.")