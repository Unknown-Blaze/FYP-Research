#!/usr/bin/env python3
"""
Test script to verify the model training setup works correctly
"""

import os
import sys
import torch

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the test function from model_training.py
from model_training import test_model_setup, setup_device

def main():
    """Test the model setup"""
    print("🧪 Testing Model Training Setup")
    print("="*50)
    
    # Test device setup
    print("🔧 Testing device setup...")
    try:
        device = setup_device()
        print(f"✅ Device setup successful: {device}")
    except Exception as e:
        print(f"❌ Device setup failed: {e}")
        return False
    
    # Test model setup
    print("\n🔧 Testing model setup...")
    try:
        success = test_model_setup()
        if success:
            print("🎉 All tests passed! The setup is ready for training.")
            return True
        else:
            print("❌ Model setup test failed!")
            return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Setup verification completed successfully!")
            print("🚀 You can now run the training script: python model_training.py")
        else:
            print("\n❌ Setup verification failed!")
            print("🔧 Please fix the issues before proceeding with training.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
