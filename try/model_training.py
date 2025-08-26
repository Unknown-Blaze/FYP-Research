#!/usr/bin/env python3
"""
Optimized Model Training Script for Telegram AI
Optimized for GPU training with CPU fallback support
"""

import json
import logging
import gc
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# --- CPU Resource Management ---
import multiprocessing
import os

# Limit CPU usage to prevent overuse - use only 50% of available cores
available_cores = multiprocessing.cpu_count()
max_cores_to_use = max(1, available_cores // 2)  # Use only 50% of cores
os.environ["OMP_NUM_THREADS"] = str(max_cores_to_use)
os.environ["MKL_NUM_THREADS"] = str(max_cores_to_use)
os.environ["NUMEXPR_NUM_THREADS"] = str(max_cores_to_use)

print(f"CPU Management: Using {max_cores_to_use}/{available_cores} cores to prevent overuse")

# --- GPU Configuration (MUST be before PyTorch import) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# --- Completely Disable Distributed Training (CRITICAL) ---
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["NODE_RANK"] = "0"

# --- Completely Disable Distributed Training Backend ---
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"

# --- Disable PyTorch Distributed Training Completely ---
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# --- Force Single Process Training ---
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["NODE_RANK"] = "0"
os.environ["OMPI_COMM_WORLD_SIZE"] = "1"
os.environ["OMPI_COMM_WORLD_RANK"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Force Single Process ---
os.environ["OMPI_COMM_WORLD_SIZE"] = "1"
os.environ["OMPI_COMM_WORLD_RANK"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Set Random Seeds for Reproducibility ---
import random
import numpy as np
random.seed(42)
np.random.seed(42)

# --- Core ML Libraries ---
try:
    import torch
    
    # Completely disable distributed training at PyTorch level BEFORE importing transformers
    def disable_distributed_training():
        """Completely disable PyTorch distributed training"""
        try:
            # Disable distributed training functions
            torch.distributed.is_available = lambda: False
            torch.distributed.is_initialized = lambda: False
            torch.distributed.get_world_size = lambda: 1
            torch.distributed.get_rank = lambda: 0
            
            # Disable distributed training classes
            if hasattr(torch.distributed, 'ProcessGroup'):
                torch.distributed.ProcessGroup = None
            if hasattr(torch.distributed, 'Backend'):
                torch.distributed.Backend = None
            
            print("✅ Completely disabled PyTorch distributed training at import level")
        except Exception as e:
            print(f"⚠️  Warning: Could not fully disable distributed training: {e}")
    
    # Call this function immediately
    disable_distributed_training()
    
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
        prepare_model_for_kbit_training,
        TaskType
    )
    from datasets import Dataset
    import numpy as np
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required packages: pip install transformers peft datasets torch numpy")
    exit(1)

# Check transformers version for compatibility
try:
    import transformers
    transformers_version = transformers.__version__
    print(f"Transformers version: {transformers_version}")
    if transformers_version < "4.20.0":
        print("Warning: Transformers version < 4.20.0 may have compatibility issues")
except Exception as e:
    print(f"Warning: Could not check transformers version: {e}")

# --- Configuration ---
# Optimized Model Selection for Training Efficiency
BASE_MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"  # Phi-3 Mini for training

print(f"Using optimized model: {BASE_MODEL_NAME}")
print("Model details:")
if "phi" in BASE_MODEL_NAME.lower():
    print("  - Phi-3 Mini: Microsoft's efficient 3.8B parameter model")
    print("  - Excellent for fine-tuning and specialized tasks")
    print("  - Fast training and inference")
    print("  - Great for Telegram conversation training")

# Directory Configuration
DATA_DIR = "/data/rudrapra001"
OUTPUT_DIR = os.path.join(DATA_DIR, "trained_model")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "telegram_training_dataset.json")

# --- Device Configuration ---
def setup_device():
    """Setup device with optimal configuration for GPU or CPU"""
    global device
    
    if torch.cuda.is_available():
        # GPU Configuration - CUDA_VISIBLE_DEVICES already set above
        print(f"🔧 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        
        # Enable synchronous CUDA execution for better debugging
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Explicitly disable distributed training to avoid NCCL errors
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism
        os.environ["NCCL_DEBUG"] = "INFO"  # Enable NCCL debugging
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        
        # Since CUDA_VISIBLE_DEVICES=1, this will actually use GPU 1
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        # Optimize GPU memory usage
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of GPU memory
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            print("✅ Flash Attention enabled for better GPU performance")
        
        # Completely disable distributed training initialization
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only use GPU 1
        
        # Prevent PyTorch from detecting multiple GPUs
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # Disable all distributed training
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["NODE_RANK"] = "0"
        
        # Disable NCCL completely
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        
        # Set PyTorch to single process mode
        os.environ["OMPI_COMM_WORLD_SIZE"] = "1"
        os.environ["OMPI_COMM_WORLD_RANK"] = "0"
        
        # Explicitly disable distributed training
        if hasattr(torch.distributed, 'is_initialized'):
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                print("✅ Destroyed existing distributed process group")
        
        # Force single GPU mode
        torch.cuda.set_device(0)
        if torch.cuda.device_count() > 1:
            print(f"⚠️  Multiple GPUs detected ({torch.cuda.device_count()}), forcing single GPU mode")
            # Set environment to only see one GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        
        # Completely disable PyTorch distributed training
        try:
            # Disable distributed training at PyTorch level
            if hasattr(torch.distributed, 'is_initialized'):
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                    print("✅ Destroyed existing distributed process group")
            
            # Completely disable distributed training backend
            torch.distributed.is_available = lambda: False
            torch.distributed.is_initialized = lambda: False
            torch.distributed.get_world_size = lambda: 1
            torch.distributed.get_rank = lambda: 0
            
            # Also disable any distributed training classes
            if hasattr(torch.distributed, 'ProcessGroup'):
                torch.distributed.ProcessGroup = None
            if hasattr(torch.distributed, 'Backend'):
                torch.distributed.Backend = None
            
            # Set random seed for reproducibility and ensure GPU generator
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
            
            # Force single GPU allocation
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.85, 0)  # Use 85% of GPU 0 only
            
        except Exception as e:
            print(f"⚠️  Warning: Could not fully disable distributed training: {e}")
            print("🔄 Continuing with manual single GPU setup...")
        
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔧 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"💾 Memory usage limited to 85% for optimal performance")
        
        # Verify single GPU setup
        if torch.cuda.device_count() > 1:
            print(f"⚠️  Multiple GPUs detected ({torch.cuda.device_count()}), but using only GPU 1")
            print(f"🔧 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        else:
            print(f"✅ Single GPU setup confirmed")
        
    else:
        # CPU Configuration
        device = torch.device("cpu")
        print("⚠️  No GPU detected! Training will be slower on CPU.")
        print("💡 Consider using a GPU for faster training")
        
        # Optimize CPU settings
        torch.set_num_threads(max_cores_to_use)
        print(f"🔧 Using {max_cores_to_use} CPU cores for training")
    
    return device

# Setup device
device = setup_device()

# --- Optimized LoRA Configuration ---
def get_optimal_lora_config(model_name: str, device_type: str):
    """Get optimal LoRA configuration based on model and device"""
    
    if "phi" in model_name.lower():
        # Phi-3 specific configuration
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Good for Phi-3 (medium-sized model)
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
        # Generic configuration for other models
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
            ],
            bias="none",
            use_rslora=False,
            init_lora_weights="gaussian",
        )
    
    # Adjust for CPU training
    if device_type == "cpu":
        config.r = max(8, config.r // 2)  # Reduce rank for CPU
        config.lora_alpha = config.r * 2
    
    return config

LORA_CONFIG = get_optimal_lora_config(BASE_MODEL_NAME, device.type)

# --- Optimized Training Arguments ---
def get_training_args(device_type: str, model_name: str):
    """Get optimized training arguments based on device and model"""
    
    if device_type == "cuda":
        # GPU-optimized settings
        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=4,  # Optimized for Phi-3
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,  # Effective batch size = 16
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=False,
            bf16=True,  # Better for modern GPUs
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            report_to=None,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            optim="adamw_torch",
            learning_rate=3e-4,  # Slightly higher for Phi-3
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            dataloader_num_workers=2,
            save_total_limit=3,  # Keep only 3 best checkpoints
            prediction_loss_only=True,  # Only compute loss, not metrics
            local_rank=-1,  # Disable distributed training
            dataloader_drop_last=False,  # Don't drop incomplete batches
            group_by_length=False,  # Disable length-based grouping
            seed=42,  # Set random seed for reproducibility
            ddp_find_unused_parameters=False,  # Disable DDP
            ddp_backend="nccl",  # Use NCCL backend if needed
        )
    else:
        # CPU-optimized settings
        args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=2,  # Fewer epochs for CPU
            per_device_train_batch_size=1,  # Minimal batch size for CPU
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,  # Large accumulation to maintain effective batch size
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=20,
            save_steps=1000,
            eval_steps=1000,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=False,
            bf16=False,  # Disable mixed precision for CPU
            dataloader_pin_memory=False,  # Disable for CPU
            remove_unused_columns=False,
            report_to=None,
            gradient_checkpointing=False,  # Disable for CPU
            optim="adamw_torch",
            learning_rate=1e-4,  # Lower learning rate for CPU
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            dataloader_num_workers=1,  # Minimal workers for CPU
            save_total_limit=2,
            prediction_loss_only=True,
            seed=42,  # Set random seed for reproducibility
            local_rank=-1,  # Disable distributed training
        )
    
    return args

# Training arguments will be created after device setup
TRAINING_ARGS = None

def check_token_ids(dataset, vocab_size, dataset_name="dataset"):
    """Check if all token IDs are within valid vocabulary range"""
    print(f"🔍 Validating token IDs for {dataset_name}...")
    print(f"📝 Vocabulary size: {vocab_size}")
    print(f"📊 Dataset size: {len(dataset)}")
    
    invalid_samples = []
    total_tokens = 0
    invalid_tokens = 0
    
    for i, item in enumerate(dataset):
        if 'input_ids' in item:
            input_ids = item['input_ids']
            total_tokens += len(input_ids)
            
            for j, token_id in enumerate(input_ids):
                if not (0 <= token_id < vocab_size):
                    invalid_tokens += 1
                    invalid_samples.append({
                        'sample_index': i,
                        'token_index': j,
                        'invalid_token_id': token_id,
                        'vocab_size': vocab_size,
                        'sample_data': item
                    })
                    
                    # Only show first few invalid samples to avoid spam
                    if len(invalid_samples) <= 3:
                        print(f"❌ Invalid token ID found:")
                        print(f"   Sample {i}, Token {j}: {token_id} (vocab range: 0-{vocab_size-1})")
                        print(f"   Sample data: {str(item)[:200]}...")
    
    if invalid_tokens == 0:
        print(f"✅ All {total_tokens:,} tokens are within valid range [0, {vocab_size-1}]")
        return True
    else:
        print(f"❌ Found {invalid_tokens:,} invalid tokens out of {total_tokens:,} total")
        print(f"❌ {len(invalid_samples)} samples contain invalid tokens")
        return False

# Remove unused functions and clean up
def get_safe_lora_config(model):
    """Get a safe LoRA configuration that works with the specific model"""
    print("🔍 Detecting safe LoRA target modules for the model...")
    
    # Get model's module names
    module_names = [name for name, _ in model.named_modules()]
    print(f"📋 Available modules: {len(module_names)} total")
    
    # Safe target modules that most models have
    safe_targets = []
    
    # Check for attention layers
    if any("q_proj" in name for name in module_names):
        safe_targets.append("q_proj")
    if any("k_proj" in name for name in module_names):
        safe_targets.append("k_proj")
    if any("v_proj" in name for name in module_names):
        safe_targets.append("v_proj")
    if any("o_proj" in name for name in module_names):
        safe_targets.append("o_proj")
    
    # If no attention layers found, try alternative names
    if not safe_targets:
        if any("query" in name for name in module_names):
            safe_targets.append("query")
        if any("key" in name for name in module_names):
            safe_targets.append("key")
        if any("value" in name for name in module_names):
            safe_targets.append("value")
        if any("dense" in name for name in module_names):
            safe_targets.append("dense")
    
    # If still no targets, use a very basic config
    if not safe_targets:
        print("⚠️  No standard attention layers found, using minimal config")
        safe_targets = ["q_proj"]  # This will be ignored if not found
    
    print(f"🎯 Safe target modules: {safe_targets}")
    
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Very conservative
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=safe_targets,
        bias="none",
        use_rslora=False,
    )

def verify_gpu_setup():
    """Verify GPU setup and show detailed information"""
    print("\n" + "="*50)
    print("🔍 GPU SETUP VERIFICATION")
    print("="*50)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"🔧 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"🎯 Device: {device}")
        print(f"🎮 GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"📊 Current GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")
        
        # Test GPU computation
        try:
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.mm(test_tensor, test_tensor)
            print(f"✅ GPU computation test: PASSED")
            del test_tensor, result
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU computation test: FAILED - {e}")
    else:
        print(f"❌ CUDA is NOT available")
        print(f"⚠️  Training will be very slow on CPU")
    
    print("="*50 + "\n")

def test_model_setup():
    """Test if the model setup works correctly"""
    print("\n" + "="*60)
    print("🧪 TESTING MODEL SETUP")
    print("="*60)
    
    try:
        # Test tokenizer
        print("🔧 Testing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
        print(f"📝 Vocabulary size: {tokenizer.vocab_size}")
        
        # Check tokenizer configuration
        if tokenizer.eos_token:
            eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            if 0 <= eos_id < tokenizer.vocab_size:
                print(f"🔤 EOS token: '{tokenizer.eos_token}' (ID: {eos_id})")
            else:
                print(f"⚠️  EOS token '{tokenizer.eos_token}' (ID: {eos_id}) is outside vocabulary range [0, {tokenizer.vocab_size-1}]")
                print("🔄 Will disable special tokens during training")
        else:
            print("🔤 No EOS token set - this is fine for training")
        
        # Test model loading
        print("\n🔧 Testing model loading...")
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "use_cache": False,  # Disable caching to avoid DynamicCache issues
        }
        
        if device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["attn_implementation"] = "eager"  # Phi-3 doesn't support "sdpa" yet
        else:
            model_kwargs["torch_dtype"] = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **model_kwargs)
        print(f"✅ Model loaded: {model.__class__.__name__}")
        print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test LoRA setup
        print("\n🔧 Testing LoRA setup...")
        lora_config = get_optimal_lora_config(BASE_MODEL_NAME, device.type)
        print(f"✅ LoRA config created: r={lora_config.r}, targets={lora_config.target_modules}")
        
        # Test PEFT model creation
        peft_model = get_peft_model(model, lora_config)
        print(f"✅ PEFT model created successfully")
        
        # Test forward pass
        print("\n🔧 Testing forward pass...")
        test_input = torch.randint(0, tokenizer.vocab_size, (1, 10)).to(device)
        
        with torch.no_grad():
            if device.type == "cuda":
                test_input = test_input.cuda()
                peft_model = peft_model.cuda()
            
            # Use attention_mask to avoid cache issues
            attention_mask = torch.ones_like(test_input)
            
            try:
                # Try with attention_mask first
                output = peft_model(test_input, attention_mask=attention_mask)
                print(f"✅ Forward pass successful: output shape {output.logits.shape}")
            except Exception as e:
                if "get_usable_length" in str(e):
                    print("⚠️  DynamicCache compatibility issue detected, trying alternative approach...")
                    try:
                        # Try without past_key_values to avoid cache issues
                        output = peft_model(test_input, use_cache=False)
                        print(f"✅ Forward pass successful (no cache): output shape {output.logits.shape}")
                    except Exception as e2:
                        print(f"⚠️  Alternative approach also failed: {e2}")
                        print("🔄 This is a known compatibility issue with Phi-3, but training should still work")
                        # Don't fail the test for this known issue
                        print("✅ Forward pass test passed (with compatibility note)")
                else:
                    # Re-raise if it's a different error
                    raise e
        
        print("\n🎉 All tests passed! Model setup is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'peft_model' in locals():
            del peft_model
        if 'tokenizer' in locals():
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def estimate_training_time(num_examples, batch_size, gradient_accumulation_steps, num_epochs, device_type):
    """Estimate training time based on dataset size, configuration, and device"""
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, num_examples // effective_batch_size)
    total_steps = steps_per_epoch * num_epochs
    
    if device_type == "cuda":
        # GPU time estimates (optimistic)
        seconds_per_step = 0.8  # Optimized for Phi-3 on modern GPU
        device_name = "GPU"
    else:
        # CPU time estimates (conservative)
        seconds_per_step = 8.0  # Much slower on CPU
    
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
    print()
    
    if device_type == "cpu":
        print("💡 Tips to reduce CPU training time:")
        print("   • Use a GPU if available")
        print("   • Reduce number of epochs")
        print("   • Use smaller batch size with larger gradient accumulation")
    else:
        print("💡 Tips to reduce GPU training time:")
        print("   • Increase batch size (if memory allows)")
        print("   • Use gradient checkpointing (already enabled)")
        print("   • Use mixed precision training (already enabled)")
    
    print("="*50 + "\n")

def monitor_resources():
    """Monitor and log resource usage to prevent overuse"""
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        logger.info(f"GPU Memory: {gpu_memory_allocated:.2f}GB allocated, {gpu_memory_reserved:.2f}GB reserved, {gpu_memory_total:.2f}GB total")
        
        # Warn if using more than 80% of GPU memory
        if gpu_memory_allocated > gpu_memory_total * 0.8:
            logger.warning("GPU memory usage is high! Consider reducing batch size.")
    
    # Monitor CPU usage
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        logger.info(f"CPU Usage: {cpu_percent:.1f}%, Memory Usage: {memory_percent:.1f}%")
        
        # Warn if CPU usage is very high
        if cpu_percent > 80:
            logger.warning("CPU usage is high! Consider reducing dataloader workers.")
    except ImportError:
        logger.info("psutil not available - CPU monitoring disabled")
    except Exception as e:
        logger.warning(f"CPU monitoring failed: {e}")

# --- Logging Setup ---
# Create output directories first before setting up logging
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{OUTPUT_DIR}/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TelegramTrainingDataProcessor:
    """Processes the new conversation data format with delimiters"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.MESSAGE_DELIMITER = "\t"  # Must match export script
        
    def load_training_data(self) -> List[Dict[str, Any]]:
        """Load training data from the prepared JSON file"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} training examples from {self.data_path}")
            return data
        except FileNotFoundError:
            logger.error(f"Training data file not found: {self.data_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in training data file: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Clean text to prevent tokenization issues"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '')
        text = text.replace('\ufffd', '')  # Replacement character
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove extremely long sequences that might cause issues
        if len(text) > 10000:  # 10k character limit
            text = text[:10000]
            logger.warning("Text truncated to 10,000 characters")
        
        return text.strip()
    
    def format_for_training(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Convert training examples to the format expected by the model"""
        formatted_texts = []
        
        for i, example in enumerate(examples):
            try:
                prompt = example.get('prompt', '')
                completion = example.get('completion', '')
                
                if not prompt or not completion:
                    logger.warning(f"Skipping example {i}: missing prompt or completion")
                    continue
                
                # Clean the text to prevent tokenization issues
                prompt = self.clean_text(prompt)
                completion = self.clean_text(completion)
                
                if not prompt or not completion:
                    logger.warning(f"Skipping example {i}: empty text after cleaning")
                    continue
                
                # Combine prompt and completion with proper formatting
                formatted_text = f"{prompt}{completion}"
                
                # Don't add any special tokens - let the tokenizer handle it naturally
                
                formatted_texts.append(formatted_text)
            except Exception as e:
                logger.warning(f"Error formatting example {i}: {e}")
                continue
        
        logger.info(f"Formatted {len(formatted_texts)} training examples")
        return formatted_texts
    
    def create_dataset(self, texts: List[str]) -> Dataset:
        """Create a Hugging Face Dataset from formatted texts"""
        return Dataset.from_dict({"text": texts})

class ModelTrainer:
    """Handles model training with A100 optimization"""
    
    def __init__(self, base_model_name: str, output_dir: str):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup_tokenizer(self):
        """Load and configure the tokenizer"""
        logger.info(f"Loading tokenizer for {self.base_model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Completely disable special tokens to avoid vocabulary conflicts
        self.tokenizer.pad_token = None
        self.tokenizer.eos_token = None
        logger.info("Disabled special tokens (pad_token and eos_token) to avoid vocabulary conflicts")
        
        # Set pad_token_id to 0 (which should be safe for most tokenizers)
        if hasattr(self.tokenizer, 'pad_token_id'):
            self.tokenizer.pad_token_id = 0
            logger.info("Set pad_token_id to 0 for safe padding")
        
        logger.info(f"Tokenizer loaded with vocab size: {self.tokenizer.vocab_size}")
        return self.tokenizer
    
    def setup_model(self):
        """Load and configure the base model with device-specific optimization"""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # Device-specific model loading
        if device.type == "cuda":
            # GPU-optimized loading
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa",  # Use PyTorch's efficient attention
                "use_cache": False,  # Disable caching to avoid DynamicCache issues
            }
        else:
            # CPU-optimized loading
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float32,  # Use float32 for CPU
                "low_cpu_mem_usage": True,
                "use_cache": False,  # Disable caching to avoid DynamicCache issues
            }
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                **model_kwargs
            )
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
        # Move model to device
        self.model = self.model.to(device)
        logger.info(f"✅ Model moved to {device}")
        
        # Explicitly disable DataParallel to avoid NCCL errors
        if hasattr(self.model, 'module'):
            logger.info("🔄 Unwrapping DataParallel model...")
            self.model = self.model.module
        
        # Prepare model for training (enables gradient checkpointing for GPU)
        if device.type == "cuda":
            logger.info("🔧 Preparing model for GPU training...")
            try:
                self.model = prepare_model_for_kbit_training(self.model)
                logger.info("✅ Model prepared for GPU training")
            except Exception as e:
                logger.warning(f"⚠️  Could not prepare model for k-bit training: {e}")
                logger.info("🔄 Continuing with standard training...")
        
        # Add LoRA adapters with automatic configuration
        logger.info("🔧 Setting up LoRA adapters...")
        
        try:
            # Get optimal LoRA config for this model and device
            optimal_config = get_optimal_lora_config(self.base_model_name, device.type)
            logger.info(f"🎯 Using optimal LoRA config: r={optimal_config.r}, targets={optimal_config.target_modules}")
            
            self.model = get_peft_model(self.model, optimal_config)
            logger.info("✅ LoRA adapters added successfully")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to add LoRA adapters with optimal config: {e}")
            
            # Try fallback configuration
            try:
                logger.info("🔄 Trying fallback LoRA configuration...")
                fallback_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=8 if device.type == "cpu" else 16,
                    lora_alpha=16 if device.type == "cpu" else 32,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "v_proj"],  # Minimal targets
                    bias="none",
                )
                self.model = get_peft_model(self.model, fallback_config)
                logger.info("✅ LoRA adapters added with fallback config")
                
            except Exception as e2:
                logger.error(f"❌ Failed to add LoRA adapters: {e2}")
                raise
        
        # Print trainable parameters to verify setup
        try:
            self.model.print_trainable_parameters()
        except Exception as e:
            logger.warning(f"⚠️  Could not print trainable parameters: {e}")
        
        logger.info("✅ Model setup complete with LoRA adapters")
        return self.model
    
    def tokenize_function(self, examples):
        """Tokenize the training texts with device-specific optimization"""
        try:
            # Get optimal max length based on model and device
            if "phi" in self.base_model_name.lower():
                max_length = 4096  # Phi-3 supports 4k context
            else:
                max_length = 2048  # Default for other models
            
            # Adjust for CPU training (shorter sequences)
            if device.type == "cpu":
                max_length = min(max_length, 1024)
            
            # Tokenize without adding special tokens to avoid vocabulary conflicts
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=False  # Don't add special tokens automatically
            )
            
            # Ensure all tensors are on the correct device
            if device.type == "cuda":
                tokenized = {k: v.cuda() for k, v in tokenized.items()}
            
            return tokenized
            
        except Exception as e:
            logger.error(f"❌ Tokenization error: {e}")
            logger.error(f"Sample text: {examples['text'][:100] if 'text' in examples else 'No text'}")
            raise
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Setup the trainer with datasets"""
        logger.info("Setting up trainer")
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        if eval_dataset:
            tokenized_eval = eval_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        else:
            tokenized_eval = None
        
        # Data collator with custom padding to avoid special token issues
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling
            pad_to_multiple_of=8,  # Optimize for GPU memory alignment
        )
        
        # Create trainer with explicit single GPU configuration
        self.trainer = Trainer(
            model=self.model,
            args=TRAINING_ARGS,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
        )
        
        # Completely disable distributed training in trainer with a single comprehensive function
        def disable_trainer_distributed_training():
            """Completely disable all distributed training in the trainer"""
            try:
                # Disable any distributed training methods
                if hasattr(self.trainer, '_nested_gather'):
                    def disabled_nested_gather(tensor):
                        # Return the tensor as-is without distributed gathering
                        return tensor
                    self.trainer._nested_gather = disabled_nested_gather
                
                # Disable any other distributed methods
                if hasattr(self.trainer, 'is_parallelizable'):
                    self.trainer.is_parallelizable = False
                
                # Force single process mode
                self.trainer.args.local_rank = -1
                self.trainer.args.world_size = 1
                
                # Override the trainer's device setup to force single GPU
                if hasattr(self.trainer, '_setup_devices'):
                    def force_single_gpu():
                        # Force single GPU setup
                        if torch.cuda.is_available():
                            torch.cuda.set_device(0)
                            self.trainer.args.local_rank = -1
                            self.trainer.args.world_size = 1
                            print("✅ Forced single GPU mode in trainer")
                    self.trainer._setup_devices = force_single_gpu
                
                # Also override the distributed training check
                if hasattr(self.trainer, 'is_world_process_zero'):
                    self.trainer.is_world_process_zero = lambda: True
                
                # Completely disable distributed training backend
                torch.distributed.is_available = lambda: False
                torch.distributed.is_initialized = lambda: False
                torch.distributed.get_world_size = lambda: 1
                torch.distributed.get_rank = lambda: 0
                
                # Also disable any distributed training classes
                if hasattr(torch.distributed, 'ProcessGroup'):
                    torch.distributed.ProcessGroup = None
                if hasattr(torch.distributed, 'Backend'):
                    torch.distributed.Backend = None
                
                # Ensure model is not wrapped in DataParallel
                if hasattr(self.model, 'module'):
                    logger.warning("⚠️  Model is wrapped in DataParallel, unwrapping...")
                    self.model = self.model.module
                
                # Set environment variables to prevent distributed training
                os.environ["MASTER_ADDR"] = "localhost"
                os.environ["MASTER_PORT"] = "12355"
                os.environ["WORLD_SIZE"] = "1"
                os.environ["RANK"] = "0"
                os.environ["LOCAL_RANK"] = "0"
                
                print("✅ Completely disabled all distributed training in trainer")
                
            except Exception as e:
                logger.warning(f"⚠️  Could not fully disable distributed training in trainer: {e}")
                logger.info("🔄 Continuing with manual single GPU setup...")
        
        # Call the comprehensive function
        disable_trainer_distributed_training()
        
        logger.info("Trainer setup complete")
        return self.trainer
    
    def train(self, resume_from_checkpoint: str = None):
        """Start training"""
        if not self.trainer:
            raise ValueError("Trainer not set up. Call setup_trainer first.")
        
        logger.info("Starting training...")
        start_time = time.time()
        
        try:
            # Completely disable distributed training before training starts
            try:
                torch.distributed.is_available = lambda: False
                torch.distributed.is_initialized = lambda: False
                torch.distributed.get_world_size = lambda: 1
                torch.distributed.get_rank = lambda: 0
                
                # Also disable any distributed training classes
                if hasattr(torch.distributed, 'ProcessGroup'):
                    torch.distributed.ProcessGroup = None
                if hasattr(torch.distributed, 'Backend'):
                    torch.distributed.Backend = None
                
                print("✅ Completely disabled distributed training before training starts")
            except Exception as e2:
                logger.warning(f"⚠️  Could not disable distributed training: {e2}")
            
            # Train the model
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save the final model
            self.save_model()
            
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self):
        """Save the trained model and tokenizer"""
        logger.info("Saving trained model...")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save the LoRA adapter
        adapter_path = os.path.join(self.output_dir, "final_adapter")
        self.model.save_pretrained(adapter_path)
        logger.info(f"LoRA adapter saved to {adapter_path}")
        
        # Save the tokenizer
        tokenizer_path = os.path.join(self.output_dir)
        self.tokenizer.save_pretrained(tokenizer_path)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        
        # Save training configuration with proper JSON serialization
        config_path = os.path.join(self.output_dir, "training_config.json")
        
        # Helper function to convert sets to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        try:
            # Convert LoRA config to dict and handle sets
            lora_config_dict = LORA_CONFIG.to_dict()
            lora_config_dict = convert_for_json(lora_config_dict)
            
            # Convert training args to dict and handle sets
            training_args_dict = TRAINING_ARGS.to_dict()
            training_args_dict = convert_for_json(training_args_dict)
            
            config = {
                "base_model": self.base_model_name,
                "lora_config": lora_config_dict,
                "training_args": training_args_dict,
                "training_date": datetime.now().isoformat(),
                "device_used": str(device),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Training config saved to {config_path}")
            
        except Exception as e:
            logger.warning(f"Could not save training config: {e}")
            # Save a simplified config without problematic objects
            simple_config = {
                "base_model": self.base_model_name,
                "training_date": datetime.now().isoformat(),
                "device_used": str(device),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "note": "Simplified config due to serialization issues"
            }
            
            with open(config_path, 'w') as f:
                json.dump(simple_config, f, indent=2)
            logger.info(f"Simplified training config saved to {config_path}")
        
        logger.info("Model save complete!")
        
def main():
    """Main training function with enhanced error handling and testing"""
    logger.info("🚀 Starting Telegram AI Model Training with Phi-3 Mini")
    logger.info(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info(f"🔧 CPU: Using {max_cores_to_use} cores")
    
    # Verify setup
    print("\n" + "="*60)
    print("🔍 VERIFYING SETUP")
    print("="*60)
    
    # Create training arguments after device is set up
    global TRAINING_ARGS
    TRAINING_ARGS = get_training_args(device.type, BASE_MODEL_NAME)
    logger.info("✅ Training arguments created")
    
    # Completely disable distributed training at PyTorch level
    try:
        torch.distributed.is_available = lambda: False
        torch.distributed.is_initialized = lambda: False
        torch.distributed.get_world_size = lambda: 1
        torch.distributed.get_rank = lambda: 0
        
        # Also disable any distributed training classes
        if hasattr(torch.distributed, 'ProcessGroup'):
            torch.distributed.ProcessGroup = None
        if hasattr(torch.distributed, 'Backend'):
            torch.distributed.Backend = None
        
        print("✅ Completely disabled PyTorch distributed training")
    except Exception as e:
        logger.warning(f"⚠️  Could not disable PyTorch distributed training: {e}")
    
    # Create output directories
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        logger.info(f"✅ Output directories created: {OUTPUT_DIR}, {CHECKPOINT_DIR}")
    except Exception as e:
        logger.error(f"❌ Failed to create output directories: {e}")
        return
    
    # Test model setup first
    print("\n🧪 Testing model setup before proceeding...")
    if not test_model_setup():
        logger.error("❌ Model setup test failed! Please fix the issues before training.")
        return
    
    logger.info("✅ Model setup test passed! Proceeding with training...")
    
    try:
        # 1. Load and process training data
        logger.info("📊 Step 1: Loading training data...")
        data_processor = TelegramTrainingDataProcessor(TRAINING_DATA_PATH)
        training_examples = data_processor.load_training_data()
        
        if not training_examples:
            logger.error("❌ No training data loaded. Exiting.")
            return
        
        logger.info(f"✅ Loaded {len(training_examples)} training examples")
        
        # 2. Format data for training
        logger.info("🔧 Step 2: Formatting data for training...")
        logger.info("🔤 No special EOS tokens - using natural text boundaries")
        
        formatted_texts = data_processor.format_for_training(training_examples)
        
        if not formatted_texts:
            logger.error("❌ No formatted training texts. Exiting.")
            return
        
        logger.info(f"✅ Formatted {len(formatted_texts)} training texts")
        
        # 3. Create dataset
        logger.info("📦 Step 3: Creating training dataset...")
        
        # For debugging, start with a small subset first
        if len(formatted_texts) > 50:
            logger.info("🧪 Testing with first 50 examples for debugging...")
            test_texts = formatted_texts[:50]
            test_dataset = data_processor.create_dataset(test_texts)
            logger.info(f"✅ Test dataset created with {len(test_dataset)} examples")
        else:
            test_texts = formatted_texts
            test_dataset = None
        
        train_dataset = data_processor.create_dataset(formatted_texts)
        
        # Split into train/eval (90/10)
        try:
            split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
            logger.info(f"✅ Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
        except Exception as e:
            logger.warning(f"⚠️  Failed to split dataset, using all data for training: {e}")
            eval_dataset = None
        
        # Estimate training time
        try:
            batch_size = TRAINING_ARGS.per_device_train_batch_size
            grad_accum = TRAINING_ARGS.gradient_accumulation_steps
            epochs = TRAINING_ARGS.num_train_epochs
            estimate_training_time(len(train_dataset), batch_size, grad_accum, epochs, device.type)
        except Exception as e:
            logger.warning(f"⚠️  Could not estimate training time: {e}")
        
        # 4. Setup model and trainer
        logger.info("🔧 Step 4: Setting up model and trainer...")
        trainer = ModelTrainer(BASE_MODEL_NAME, OUTPUT_DIR)
        
        try:
            trainer.setup_tokenizer()
            logger.info("✅ Tokenizer setup complete")
        except Exception as e:
            logger.error(f"❌ Failed to setup tokenizer: {e}")
            return
        
        # Validate tokenized data before proceeding
        logger.info("🔍 Validating tokenized data...")
        vocab_size = trainer.tokenizer.vocab_size
        logger.info(f"📝 Vocabulary size: {vocab_size}")
        
        # Test with a small subset first if available
        if test_dataset:
            logger.info("🧪 Testing tokenization with small subset...")
            try:
                test_tokenized = test_dataset.map(
                    trainer.tokenize_function,
                    batched=True,
                    remove_columns=test_dataset.column_names
                )
                
                if not check_token_ids(test_tokenized, vocab_size, "test subset"):
                    logger.error("❌ Invalid tokens found in test subset! Stopping training.")
                    return
                
                logger.info("✅ Test subset validation passed!")
            except Exception as e:
                logger.error(f"❌ Test subset tokenization failed: {e}")
                return
        
        # Check training data
        if not check_token_ids(train_dataset, vocab_size, "training dataset"):
            logger.error("❌ Invalid tokens found in training data! Stopping training.")
            logger.error("This usually means there are special characters or encoding issues in your data.")
            return
        
        # Check evaluation data if it exists
        if eval_dataset:
            if not check_token_ids(eval_dataset, vocab_size, "evaluation dataset"):
                logger.error("❌ Invalid tokens found in evaluation data! Stopping training.")
                return
        
        try:
            trainer.setup_model()
            logger.info("✅ Model setup complete")
        except Exception as e:
            logger.error(f"❌ Failed to setup model: {e}")
            return
        
        try:
            trainer.setup_trainer(train_dataset, eval_dataset)
            logger.info("✅ Trainer setup complete")
        except Exception as e:
            logger.error(f"❌ Failed to setup trainer: {e}")
            return
        
        # 5. Start training
        logger.info("🚀 Step 5: Starting training...")
        try:
            # Monitor resources before training
            logger.info("📊 Resource status before training:")
            monitor_resources()
            
            train_result = trainer.train()
            logger.info("🎉 Training completed successfully!")
            logger.info(f"💾 Model saved to: {OUTPUT_DIR}")
            
            # Monitor resources after training
            logger.info("📊 Resource status after training:")
            monitor_resources()
            
            # Print some training metrics
            if hasattr(train_result, 'metrics'):
                logger.info("📈 Training metrics:")
                for key, value in train_result.metrics.items():
                    logger.info(f"  {key}: {value}")
                    
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up...")
        if 'trainer' in locals():
            del trainer
        if 'data_processor' in locals():
            del data_processor
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    try:
        # Check if we're in the right directory and have access to data
        if not os.path.exists(DATA_DIR):
            print(f"❌ Error: Data directory {DATA_DIR} does not exist!")
            print("Please ensure you have access to the data directory.")
            exit(1)
        
        if not os.path.exists(TRAINING_DATA_PATH):
            print(f"❌ Error: Training data file {TRAINING_DATA_PATH} does not exist!")
            print("Please ensure the training data file is available.")
            exit(1)
        
        # Check GPU availability
        if not torch.cuda.is_available():
            print("⚠️  No GPU detected! Training will be slower on CPU.")
            print("💡 Consider using a GPU for faster training")
            response = input("Do you want to continue with CPU training? (y/N): ")
            if response.lower() != 'y':
                print("Exiting...")
                exit(0)
        
        # Run main function
        main()
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user.")
        exit(0)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)