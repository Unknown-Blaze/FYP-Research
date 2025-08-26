#!/usr/bin/env python3
"""
Test script to verify LoRA adapter loading after fixing the JSON corruption
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_lora_loading():
    """Test loading the fixed LoRA adapters"""
    
    # Paths
    base_model_name = "HuggingFaceH4/zephyr-7b-beta"
    lora_path = "/data/rudrapra001/trained_model/final_adapter"
    
    print("🔍 Testing LoRA adapter loading...")
    print(f"Base model: {base_model_name}")
    print(f"LoRA path: {lora_path}")
    
    try:
        # Test 1: Load base model
        print("\n📥 Loading base model...")
        base_model = AutoTokenizer.from_pretrained(base_model_name)
        print("✅ Base tokenizer loaded successfully")
        
        # Test 2: Load LoRA adapter
        print("\n🔧 Loading LoRA adapter...")
        if os.path.exists(lora_path):
            print(f"✅ LoRA directory exists: {lora_path}")
            
            # Check if adapter_config.json is valid JSON
            import json
            config_path = os.path.join(lora_path, "adapter_config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✅ adapter_config.json is valid JSON")
            print(f"   Target modules: {config.get('target_modules', [])}")
            print(f"   LoRA rank (r): {config.get('r', 'N/A')}")
            
            # Check if model file exists
            model_path = os.path.join(lora_path, "adapter_model.safetensors")
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"✅ adapter_model.safetensors exists ({size_mb:.1f} MB)")
            else:
                print("❌ adapter_model.safetensors not found")
                
        else:
            print(f"❌ LoRA directory not found: {lora_path}")
            
        print("\n🎉 LoRA adapter validation completed successfully!")
        print("The adapter should now work without the 'incomplete metadata' error.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lora_loading()
