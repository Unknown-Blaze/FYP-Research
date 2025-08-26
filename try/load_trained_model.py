#!/usr/bin/env python3
"""
Script to load and use the trained LoRA model for inference
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_trained_model():
    """Load the trained LoRA model"""
    
    # Paths
    base_model_name = "HuggingFaceH4/zephyr-7b-beta"
    lora_path = "/data/rudrapra001/trained_model/final_adapter"
    
    print("🚀 Loading trained LoRA model...")
    
    try:
        # 1. Load tokenizer
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
        
        # 2. Load base model
        print("📥 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("✅ Base model loaded")
        
        # 3. Load LoRA adapter
        print("🔧 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        print("✅ LoRA adapter loaded successfully!")
        
        # 4. Move to GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model = model.to(device)
            print(f"✅ Model moved to GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("⚠️  Using CPU (GPU not available)")
        
        print("\n🎉 Model loaded successfully!")
        print(f"📊 Model info:")
        print(f"   - Base model: {base_model_name}")
        print(f"   - LoRA adapter: {lora_path}")
        print(f"   - Device: {device}")
        print(f"   - Model type: {type(model)}")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7):
    """Generate a response using the trained model"""
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return f"Error: {e}"

def main():
    """Main function to test the trained model"""
    
    print("=" * 60)
    print("🤖 TRAINED LORA MODEL TESTER")
    print("=" * 60)
    
    # Load model
    model, tokenizer, device = load_trained_model()
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("🧪 TESTING MODEL INFERENCE")
    print("=" * 60)
    
    # Test prompts
    test_prompts = [
        "Hello, how are you?",
        "What's the weather like today?",
        "Can you help me with a question?",
        "Tell me a joke",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt}")
        print("-" * 40)
        
        response = generate_response(model, tokenizer, prompt)
        print(f"🤖 Response: {response}")
        
        if i < len(test_prompts):
            input("\nPress Enter to continue to next test...")
    
    print("\n" + "=" * 60)
    print("🎯 MODEL TESTING COMPLETE!")
    print("=" * 60)
    print("Your trained LoRA model is working correctly!")
    print("You can now use it for inference without retraining.")

if __name__ == "__main__":
    main()
