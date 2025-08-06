import torch
from transformers import (
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer,SFTConfig
import logging
import os
import time
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
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.config['base_model'],
        #     quantization_config=bnb_config,
        #     device_map="auto"
        # )
        # self.model.config.use_cache = False
        
        # self.logger.info("Loading tokenizer.")
        # self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'])

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            self.config['base_model'],
            max_seq_length=8192,
            dtype=None,
            load_in_4bit=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        peft_config = LoraConfig(**self.config['peft_config'])
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        output_model_path = os.path.join(self.config['output_dir'], self.config['run_name'])
        training_args_config = self.config['training_args']
        training_args_config['output_dir'] = output_model_path
        sftconfig = SFTConfig(**training_args_config)
        sftconfig.dataset_text_field = "text"
        return peft_config, sftconfig, output_model_path

    def train(self, train_dataset, validation_dataset):
        """Runs the fine-tuning process."""
        peft_config, training_arguments, output_model_path = self._setup_components()
        self.logger.info("Initializing SFTTrainer...")
        self.model = prepare_model_for_kbit_training(self.model)
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=self.tokenizer,
            args=training_arguments,
        )
        
        self.logger.info("Starting training...")
        start_time = time.time()

    # This is the main training step
        trainer.train()

        # --- NEW: Capture end time and calculate duration ---
        end_time = time.time()
        training_duration_seconds = end_time - start_time
        training_duration_minutes = training_duration_seconds / 60
        
        # --- NEW: Log detailed results ---
        self.logger.info("--- Training Run Summary ---")
        self.logger.info(f"Training completed in {training_duration_seconds:.2f} seconds ({training_duration_minutes:.2f} minutes).")
        final_log = trainer.state.log_history[-1]
        self.logger.info(f"Final training loss: {final_log.get('loss', 'N/A')}")
        self.logger.info(f"Final validation loss: {final_log.get('eval_loss', 'N/A')}")

        self.logger.info(f"Training complete. Saving final model to: {output_model_path}")
        trainer.save_model(output_model_path)