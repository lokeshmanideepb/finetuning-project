from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch

# Model - TinyLlama is small enough for MacBook M3
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load a small dataset like Alpaca or use dummy dataset for demo
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1%]")

# Preprocessing
def preprocess(example):
    prompt = example['instruction'] + "\n" + example['input']
    completion = example['output']
    return {
        "text": f"{prompt}\n{completion}"
    }

dataset = dataset.map(preprocess)
#dataset = dataset.remove_columns(set(dataset.features) - {"text"})

# Load base model on MPS
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map={"": "mps"})

# LoRA Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Adjust per model's architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training Arguments
training_args = SFTConfig(
    output_dir="./tiny-llama-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    optim="adamw_torch",
    remove_unused_columns=False,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

trainer.train()
