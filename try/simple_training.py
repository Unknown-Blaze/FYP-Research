#!/usr/bin/env python3
"""
Enhanced Model Training Script for Telegram AI
Fine-tune Phi-3 to speak like you - works on A100 for training, CPU for inference
"""

import json
import logging
import gc
import time
import os
import multiprocessing
from datetime import datetime
from typing import List, Dict, Any

# GPU Configuration - Set to use specific GPU and control memory growth
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change this to use different GPU (0, 1, 2, etc.)

print("CPU Management: Simplified configuration")

# Core imports
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
from datasets import Dataset

# Configuration
BASE_MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
DATA_DIR = "/data/rudrapra001"
OUTPUT_DIR = os.path.join(DATA_DIR, "trained_model")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "telegram_training_dataset.json")

# Setup logging
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{OUTPUT_DIR}/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_device():
    """Setup device with optimal configuration"""
    # Configure PyTorch to use specific GPU and control memory growth
    if torch.cuda.is_available():
        # Set the device
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        
        # Control memory growth to prevent overuse
        torch.cuda.empty_cache()
        
        # Optional: Set memory fraction (uncomment if needed)
        # torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device

def load_training_data():
    """Load and validate training data"""
    try:
        with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate data structure
        valid_examples = []
        for i, example in enumerate(data):
            if 'prompt' in example and 'completion' in example:
                if example['prompt'] and example['completion']:
                    valid_examples.append(example)
                else:
                    logger.warning(f"Example {i}: Empty prompt or completion")
            else:
                logger.warning(f"Example {i}: Missing prompt or completion fields")
        
        logger.info(f"✅ Loaded {len(valid_examples)} valid training examples from {len(data)} total")
        return valid_examples
        
    except Exception as e:
        logger.error(f"❌ Failed to load training data: {e}")
        return []

def clean_text(text: str) -> str:
    """Clean text to prevent tokenization issues"""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove null bytes and problematic characters
    text = text.replace('\x00', '')
    text = text.replace('\ufffd', '')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove extremely long sequences
    if len(text) > 10000:
        text = text[:10000]
        logger.warning("Text truncated to 10,000 characters")
    
    return text.strip()

def format_data_for_training(examples):
    """Format data for training with proper conversation structure"""
    formatted_texts = []
    
    for i, example in enumerate(examples):
        try:
            prompt = clean_text(example.get('prompt', ''))
            completion = clean_text(example.get('completion', ''))
            
            if not prompt or not completion:
                logger.warning(f"Skipping example {i}: empty text after cleaning")
                continue
            
            # Combine prompt and completion - this creates the training sequence
            # The model learns to predict the completion given the prompt
            formatted_text = f"{prompt}{completion}"
            formatted_texts.append(formatted_text)
            
        except Exception as e:
            logger.warning(f"Error formatting example {i}: {e}")
            continue
    
    logger.info(f"✅ Formatted {len(formatted_texts)} training examples")
    return formatted_texts

def get_optimal_lora_config(device_type: str):
    """Get optimal LoRA configuration for Phi-3"""
    if device_type == "cuda":
        # GPU-optimized LoRA
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Good for Phi-3
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"  # Phi-3 specific layers
            ],
            bias="none",
            use_rslora=False,
            init_lora_weights="gaussian",
        )
    else:
        # CPU-optimized LoRA (smaller)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # Smaller for CPU
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
            ],
            bias="none",
            use_rslora=False,
            init_lora_weights="gaussian",
        )
    
    return config

def get_training_arguments(device_type: str, output_dir: str):
    """Get optimized training arguments based on device"""
    if device_type == "cuda":
        # A100-optimized settings - OPTIMIZED FOR SPEED
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,  # Reduced to 1 epoch for faster training
            per_device_train_batch_size=2,  # Reduced batch size to prevent OOM
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,  # Effective batch size = 16
            warmup_steps=50,  # Reduced warmup
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=25,  # More frequent logging
            save_steps=200,  # Less frequent saving
            eval_steps=100,  # More frequent evaluation
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,  # Better for A100
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            report_to=None,
            gradient_checkpointing=False,  # Disabled for speed (A100 has enough memory)
            optim="adamw_torch",
            learning_rate=5e-4,  # Higher learning rate for faster convergence
            lr_scheduler_type="linear",  # Linear schedule for faster convergence
            warmup_ratio=0.05,  # Reduced warmup ratio
            max_grad_norm=1.0,
            dataloader_num_workers=4,  # More workers for faster data loading
            save_total_limit=2,  # Keep fewer checkpoints
            prediction_loss_only=True,
            seed=42,
        )
    else:
        # CPU-optimized settings
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,  # Fewer epochs for CPU
            per_device_train_batch_size=1,  # Minimal batch size
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,  # Large accumulation
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=20,
            save_steps=1000,
            eval_steps=1000,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=False,  # Disable for CPU
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,
            gradient_checkpointing=False,  # Disable for CPU
            optim="adamw_torch",
            learning_rate=1e-4,  # Lower for CPU
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            dataloader_num_workers=1,
            save_total_limit=2,
            prediction_loss_only=True,
            seed=42,
        )
    
    return args

def estimate_training_time(num_examples, batch_size, gradient_accumulation_steps, num_epochs, device_type):
    """Estimate training time"""
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, num_examples // effective_batch_size)
    total_steps = steps_per_epoch * num_epochs
    
    if device_type == "cuda":
        seconds_per_step = 0.4  # More realistic for A100 with optimized settings
        device_name = "A100 GPU"
    else:
        seconds_per_step = 8.0  # Conservative for CPU
    
    total_seconds = total_steps * seconds_per_step
    hours = total_seconds / 3600
    minutes = (total_seconds % 3600) / 60
    
    print("\n" + "="*50)
    print("⏰ TRAINING TIME ESTIMATION")
    print("="*50)
    print(f"📊 Dataset: {num_examples:,} examples")
    print(f"📦 Effective batch size: {effective_batch_size}")
    print(f"🔄 Steps per epoch: {steps_per_epoch:,}")
    print(f"🎯 Total training steps: {total_steps:,}")
    print(f"🖥️  Device: {device_name}")
    print(f"🧠 Model: Phi-3 Mini (3.8B parameters) with LoRA")
    print()
    print(f"⏱️  Estimated training time:")
    if hours >= 1:
        print(f"   🕐 {hours:.1f} hours ({hours*60:.0f} minutes)")
    else:
        print(f"   🕐 {minutes:.0f} minutes")
    print("="*50 + "\n")

def monitor_resources():
    """Monitor resource usage"""
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        logger.info(f"GPU Memory: {gpu_memory_allocated:.2f}GB allocated, {gpu_memory_reserved:.2f}GB reserved, {gpu_memory_total:.2f}GB total")
        
        if gpu_memory_allocated > gpu_memory_total * 0.8:
            logger.warning("GPU memory usage is high! Consider reducing batch size.")

def main():
    """Main training function"""
    logger.info("🚀 Starting Enhanced Phi-3 Fine-tuning for Personal Bot")
    logger.info(f"🎯 Goal: Make the bot speak like you based on your chat data")
    
    # Setup device
    device = setup_device()
    logger.info(f"🖥️  Device: {device}")
    
    # Load training data
    logger.info("📊 Loading training data...")
    training_examples = load_training_data()
    if not training_examples:
        logger.error("❌ No training data loaded. Exiting.")
        return
    
    # Format data
    logger.info("🔧 Formatting data for training...")
    formatted_texts = format_data_for_training(training_examples)
    if not formatted_texts:
        logger.error("❌ No formatted training texts. Exiting.")
        return
    
    # Create dataset
    logger.info("📦 Creating training dataset...")
    dataset = Dataset.from_dict({"text": formatted_texts})
    
    # Split into train/eval (90/10)
    try:
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        logger.info(f"✅ Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
    except Exception as e:
        logger.warning(f"⚠️  Failed to split dataset, using all data for training: {e}")
        eval_dataset = None
    
    # Get training arguments
    training_args = get_training_arguments(device.type, OUTPUT_DIR)
    
    # Estimate training time
    try:
        batch_size = training_args.per_device_train_batch_size
        grad_accum = training_args.gradient_accumulation_steps
        epochs = training_args.num_train_epochs
        estimate_training_time(len(train_dataset), batch_size, grad_accum, epochs, device.type)
    except Exception as e:
        logger.warning(f"⚠️  Could not estimate training time: {e}")
    
    # Load tokenizer
    logger.info("🔧 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    # Handle special tokens properly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    logger.info(f"✅ Tokenizer loaded with vocab size: {tokenizer.vocab_size}")
    
    # Load model
    logger.info("🔧 Loading Phi-3 model...")
    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",  # Phi-3 only supports eager
    }
    
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **model_kwargs)
    model = model.to(device)
    logger.info(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup LoRA
    logger.info("🔧 Setting up LoRA adapters...")
    lora_config = get_optimal_lora_config(device.type)
    logger.info(f"🎯 LoRA config: r={lora_config.r}, targets={lora_config.target_modules}")
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("✅ LoRA setup complete")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=2048,  # Conservative for memory
            return_tensors="pt",
        )
    
    # Tokenize datasets
    logger.info("🔧 Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    if eval_dataset:
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
    else:
        tokenized_eval = None
    
    logger.info("✅ Tokenization complete")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
    )
    
    # Create trainer
    logger.info("🔧 Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    logger.info("✅ Trainer created")
    
    # Monitor resources before training
    logger.info("📊 Resource status before training:")
    monitor_resources()
    
    # Train
    logger.info("🚀 Starting training...")
    start_time = time.time()
    
    try:
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        logger.info(f"🎉 Training completed successfully in {training_time:.2f} seconds!")
        
        # Save model
        logger.info("💾 Saving fine-tuned model...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Save training config
        config_path = os.path.join(OUTPUT_DIR, "training_config.json")
        
        # Convert LoRA config to JSON-serializable format
        def convert_lora_config_to_dict(lora_config):
            """Convert LoRA config to JSON-serializable dictionary"""
            config_dict = {}
            for key, value in lora_config.to_dict().items():
                if isinstance(value, set):
                    config_dict[key] = list(value)  # Convert set to list
                elif isinstance(value, (int, float, str, bool, type(None))):
                    config_dict[key] = value
                elif isinstance(value, list):
                    # Handle nested lists
                    config_dict[key] = [str(item) if not isinstance(item, (int, float, str, bool, type(None))) else item for item in value]
                else:
                    config_dict[key] = str(value)  # Convert other types to string
            return config_dict
        
        config = {
            "base_model": BASE_MODEL_NAME,
            "lora_config": convert_lora_config_to_dict(lora_config),
            "training_args": training_args.to_dict(),
            "training_date": datetime.now().isoformat(),
            "device_used": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "training_time_seconds": training_time,
            "dataset_size": len(train_dataset),
            "eval_size": len(eval_dataset) if eval_dataset else 0,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Model and config saved to {OUTPUT_DIR}")
        
        # Print training metrics
        if hasattr(train_result, 'metrics'):
            logger.info("📈 Training metrics:")
            for key, value in train_result.metrics.items():
                logger.info(f"  {key}: {value}")
        
        # Monitor resources after training
        logger.info("📊 Resource status after training:")
        monitor_resources()
        
        logger.info("🎯 Your personal bot is ready! The model can now be used on both GPU and CPU.")
        logger.info("💡 For CPU inference, just load the saved model without LoRA adapters.")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user.")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
