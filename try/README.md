# Optimized Model Training Script

This is an optimized version of the model training script for training Phi-3 Mini on both GPU and CPU.

## 🚀 Key Improvements

### GPU Optimization
- **Efficient Memory Usage**: Uses 85% of GPU memory for optimal performance
- **Flash Attention**: Enables PyTorch's efficient attention implementation
- **Mixed Precision**: Uses bfloat16 for faster training on modern GPUs
- **Gradient Checkpointing**: Reduces memory usage during training
- **Optimized Batch Sizes**: Configured specifically for Phi-3 Mini

### CPU Compatibility
- **CPU-Optimized Settings**: Automatically adjusts for CPU training
- **Reduced Memory Usage**: Smaller batch sizes with larger gradient accumulation
- **Efficient LoRA**: Reduced rank for CPU training
- **Proper Threading**: Uses optimal number of CPU cores

### Model Selection
- **Phi-3 Mini**: 3.8B parameter model for fast training and specialization
- **Efficient Architecture**: Microsoft's optimized transformer model
- **4K Context**: Supports longer sequences for better training
- **Great for Specialization**: Excellent for fine-tuning on specific tasks

## 📋 Requirements

```bash
pip install transformers peft datasets torch numpy
```

## 🔧 Usage

### 1. Test the Setup First
```bash
python test_setup.py
```

This will verify that:
- Device detection works correctly
- Model can be loaded
- LoRA adapters can be added
- Forward pass works

### 2. Run Training
```bash
python model_training.py
```

## ⚙️ Configuration

### GPU Training (Default)
- Batch size: 4
- Gradient accumulation: 4 (effective batch size: 16)
- Epochs: 3
- Learning rate: 3e-4
- Mixed precision: bfloat16

### CPU Training
- Batch size: 1
- Gradient accumulation: 16 (effective batch size: 16)
- Epochs: 2
- Learning rate: 1e-4
- Mixed precision: disabled

## 🎯 Model Details

**Phi-3 Mini (3.8B parameters)**
- **Architecture**: Transformer with grouped-query attention
- **Context Length**: 4,096 tokens
- **Training**: Pre-trained on web data
- **License**: MIT (commercial use allowed)
- **Specialization**: Great for fine-tuning on specific tasks

## 📊 Expected Performance

### GPU Training
- **Time**: ~0.8 seconds per step
- **Memory**: ~8-12 GB VRAM
- **Speed**: Fast training with mixed precision

### CPU Training
- **Time**: ~8 seconds per step
- **Memory**: ~4-6 GB RAM
- **Speed**: Slower but functional for small datasets

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `get_training_args()`
   - Enable gradient checkpointing
   - Use smaller model

2. **LoRA Configuration Errors**
   - The script automatically detects safe target modules
   - Falls back to minimal configuration if needed

3. **Tokenization Errors**
   - Check for special characters in training data
   - Verify vocabulary size compatibility

### Debug Mode
The script includes extensive logging and validation:
- Token ID validation
- Model setup testing
- Resource monitoring
- Step-by-step progress tracking

## 📁 Output

Training outputs:
- **Model**: LoRA adapters in `final_adapter/`
- **Tokenizer**: Saved in output directory
- **Logs**: Training logs in `logs/`
- **Checkpoints**: Best models saved during training
- **Config**: Training configuration saved as JSON

## 🚀 Next Steps

After training:
1. **Test the model**: Load and test with sample inputs
2. **Inference**: Use the trained LoRA adapters for generation
3. **Deployment**: Integrate into your application
4. **Iteration**: Fine-tune further based on results

## 💡 Tips

- **Start Small**: Test with 50 examples first
- **Monitor Resources**: Watch GPU/CPU usage during training
- **Save Checkpoints**: Keep intermediate models for comparison
- **Validate Data**: Ensure training data is properly formatted
- **Use GPU**: CPU training is much slower, use GPU when possible
