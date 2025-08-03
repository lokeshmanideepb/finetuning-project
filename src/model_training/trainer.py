import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer
import logging
import os

class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger()

    def _setup_components(self):
        """Initializes all components needed for training."""
        self.logger.info("Setting up training components...")
        
        quant_config = self.config.get('quantization_config', {})
        quant_config['bnb_4bit_compute_dtype'] = torch.float16 if quant_config.get('bnb_4bit_compute_dtype') == 'torch.float16' else torch.bfloat16
        bnb_config = BitsAndBytesConfig(**quant_config) if quant_config else None
        
        self.logger.info(f"Loading base model: {self.config['base_model']}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.config.use_cache = False
        
        self.logger.info("Loading tokenizer.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'], trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        peft_config = LoraConfig(**self.config['peft_config'])
        
        output_model_path = os.path.join(self.config['output_dir'], self.config['run_name'])
        training_args_config = self.config['training_args']
        training_args_config['output_dir'] = output_model_path
        training_arguments = TrainingArguments(**training_args_config)

        return peft_config, training_arguments, output_model_path

    def train(self, dataset):
        """Runs the fine-tuning process."""
        peft_config, training_arguments, output_model_path = self._setup_components()
        
        self.logger.info("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=2048,  # Ensure this matches your dataset's max length
            tokenizer=self.tokenizer,
            args=training_arguments,
            packing=False,
        )
        
        self.logger.info("Starting training...")
        trainer.train()
        
        self.logger.info(f"Training complete. Saving final model to: {output_model_path}")
        trainer.save_model(output_model_path)
