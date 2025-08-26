#!/usr/bin/env python3
"""
Download Phi-3 model for training
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_phi3():
    """Download Phi-3 model and tokenizer"""
    print("🚀 Downloading Phi-3 Mini model for training...")
    
    try:
        # Download tokenizer
        print("📝 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-3-mini-4k-instruct')
        print("✅ Tokenizer downloaded successfully!")
        print(f"   Vocabulary size: {tokenizer.vocab_size}")
        print(f"   EOS token: '{tokenizer.eos_token}'")
        
        # Download model
        print("🧠 Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            'microsoft/phi-3-mini-4k-instruct',
            torch_dtype='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Model downloaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Model type: {type(model).__name__}")
        
        print("\n🎉 Phi-3 model download complete!")
        print("💡 You can now run the training script: python model_training.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    download_phi3()
