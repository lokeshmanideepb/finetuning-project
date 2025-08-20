import logging
import json
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel

class ModelEvaluator:
    """Encapsulates the logic for evaluating a fine-tuned model using Unsloth."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger()
        self.model = None
        self.tokenizer = None

    def _load_model_and_tokenizer(self):
        """
        Loads the base model and applies the fine-tuned LoRA adapters
        using the Unsloth FastLanguageModel API.
        """
        adapter_path = self.config['adapter_path']
        
        self.logger.info(f"Loading Unsloth model and adapter from: {adapter_path}")
        
        # Unsloth's API combines loading the base model and the adapter.
        # It automatically detects the base model from the adapter's config.
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.config['base_model_path'],
            max_seq_length = 8192,
            device_map = "auto",
            # No offload_folder needed as Unsloth manages memory more efficiently
        )
        
        self.logger.info("Model and tokenizer loaded successfully with Unsloth.")
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
        instruction_template = """
            Analyze the following clinical note and identify ALL ICD-10-CM F-codes (psychiatric codes) mentioned throughout the patient's psychiatric history and current presentation. Find all temporal diagnostic changes from the patient's timeline.

            IMPORTANT RULES:
            1. Include ALL psychiatric diagnoses from patient's history (current AND historical)
            2. Temporal diagnostic changes CAN include different F-codes across timepoints - this is normal for patient timelines
            3. MAXIMUM 10 F-codes only - stop at 10 codes
            4. ONLY include F-codes (psychiatric diagnoses starting with F) - no other ICD codes
            5. For each F-code, provide ONE direct quote from the clinical note as evidence

            EXAMPLES OF TEMPORAL DIAGNOSTIC CHANGES (Normal for patient timelines):
            1. Depression Episodes: F32.A + F33.2 (single + recurrent episodes can appear across timepoints)
            2. Anxiety Disorders: F41.0 + F41.1 + F41.9 (multiple anxiety subtypes can appear across timepoints)
            3. Substance Use: F11.10 + F11.20 + F11.90 (multiple severity levels same substance can appear across timepoints)
            4. Multiple Single Depression: F32.1 + F32.9 + F32.A (multiple single episode types can appear across timepoints)
            5. Multiple Recurrent Depression: F33.1 + F33.2 (multiple recurrent episode types can appear across timepoints)
            6. Trauma/Stress: F43.10 + F43.9 (multiple PTSD/adjustment types can appear across timepoints)
            7. Bipolar Episodes: F31.30 + F31.9 (multiple bipolar episode states can appear across timepoints)
            8. ADHD Subtypes: F90.1 + F90.2 (multiple ADHD presentations can appear across timepoints)
            9. Schizoaffective: F25.0 + F25.9 (multiple schizoaffective subtypes can appear across timepoints)
            10. Schizophrenia: F20.0 + F20.9 (multiple schizophrenia subtypes can appear across timepoints)

            Format your response as:
            CODES: [list MAXIMUM 10 F-codes only, separated by commas]
            EVIDENCE: [For each F-code, provide: F32.9: "Patient diagnosed with major depressive disorder single episode" F33.2: "History of recurrent major depression with severe episodes" F41.9: "Patient exhibits anxiety symptoms with panic attacks"]
            """
        self.logger.info("Generating predictions on the test set with Unsloth model...")
        FastLanguageModel.for_inference(self.model)
        for example in tqdm(test_dataset):
            # The prompt formatting and generation loop remains the same
            prompt_text = example['prompt'] # Parse the inner JSON string
            prompt = prompt_template.format(
                instruction=instruction_template,
                input=prompt_text['input']
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_params)
            
            prediction_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_only = prediction_text.split("### Response:")[1].strip()
            ground_truth = prompt_text['output']
            
            results.append({
                "patient_id": example['patient_id'],
                "input_prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": response_only
            })

        output_path = self.config['predictions_output_path']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.logger.info(f"Saving {len(results)} predictions to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info("Evaluation run finished successfully.")