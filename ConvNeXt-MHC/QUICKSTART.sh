#!/bin/bash
# QuickStart Script for ConvNeXt-MHC Pipeline
# Run this after reading SETUP_GUIDE.md

set -e  # Exit on error

echo "=== ConvNeXt-MHC Quick Setup ==="
echo ""

# Get current directory
CONVNEXT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Working directory: $CONVNEXT_DIR"

# Step 1: Check if conda environment exists
echo ""
echo "Step 1: Checking conda environment..."
if conda env list | grep -q "convnext-mhc"; then
    echo "✓ Environment 'convnext-mhc' already exists"
else
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml
    echo "✓ Environment created"
fi

# Step 2: Setup netMHCpan
echo ""
echo "Step 2: Setting up netMHCpan-4.1..."
cd "$CONVNEXT_DIR"

if [ ! -d "netMHCpan-4.1" ]; then
    if [ -f "netMHCpan-4.1b.Linux.tar.gz" ]; then
        echo "Extracting netMHCpan-4.1b.Linux.tar.gz..."
        tar -xzf netMHCpan-4.1b.Linux.tar.gz
        echo "✓ Extracted"
    else
        echo "❌ Error: netMHCpan-4.1b.Linux.tar.gz not found!"
        echo "   Download from: https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/"
        exit 1
    fi
fi

# Step 3: Configure netMHCpan NMHOME
echo ""
echo "Step 3: Configuring netMHCpan..."
NETMHC_DIR="$CONVNEXT_DIR/netMHCpan-4.1"

if [ -f "$NETMHC_DIR/netMHCpan" ]; then
    # Backup original
    if [ ! -f "$NETMHC_DIR/netMHCpan.bak" ]; then
        cp "$NETMHC_DIR/netMHCpan" "$NETMHC_DIR/netMHCpan.bak"
    fi
    
    # Update NMHOME path
    sed -i "s|setenv.*NMHOME.*|setenv NMHOME $NETMHC_DIR|g" "$NETMHC_DIR/netMHCpan"
    echo "✓ Updated NMHOME to: $NETMHC_DIR"
else
    echo "❌ Error: netMHCpan script not found!"
    exit 1
fi

# Step 4: Create required directories
echo ""
echo "Step 4: Creating required directories..."
mkdir -p "$NETMHC_DIR/tmp"
echo "✓ Created tmp directory"

# Step 5: Copy Gen9mer.py to netMHCpan directory
echo ""
echo "Step 5: Setting up Gen9mer.py..."
if [ -f "$CONVNEXT_DIR/Gen9mer.py" ]; then
    cp "$CONVNEXT_DIR/Gen9mer.py" "$NETMHC_DIR/"
    echo "✓ Copied Gen9mer.py to netMHCpan-4.1/"
else
    echo "❌ Warning: Gen9mer.py not found at $CONVNEXT_DIR"
fi

# Step 6: Test netMHCpan installation
echo ""
echo "Step 6: Testing netMHCpan installation..."
cd "$NETMHC_DIR"
if ./netMHCpan -h > /dev/null 2>&1; then
    echo "✓ netMHCpan working correctly"
else
    echo "❌ Warning: netMHCpan test failed. Check dependencies."
fi

# Step 7: Check model weights
echo ""
echo "Step 7: Checking model weights..."
if [ -f "$CONVNEXT_DIR/ConvNeXt-MHC/save_model_weight/ConvNeXt-MHC_BA.h5" ]; then
    echo "✓ BA model found"
else
    echo "❌ Warning: ConvNeXt-MHC_BA.h5 not found!"
fi

if [ -f "$CONVNEXT_DIR/ConvNeXt-MHC/save_model_weight/ConvNeXt-MHC_AP.h5" ]; then
    echo "✓ AP model found"
else
    echo "❌ Warning: ConvNeXt-MHC_AP.h5 not found!"
fi

# Summary
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Activate environment:  conda activate convnext-mhc"
echo "2. Prepare your data CSV with columns: allele, peptide, 9mer"
echo "3. Convert to 9-mers (if needed):"
echo "   cd $NETMHC_DIR"
echo "   python Gen9mer.py your_data.csv"
echo ""
echo "4. Run predictions:"
echo "   cd $CONVNEXT_DIR/ConvNeXt-MHC"
echo "   python run_model.py"
echo ""
echo "See SETUP_GUIDE.md for detailed instructions."
echo ""

# Optional: Test with example data
read -p "Run test prediction on example.csv? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running test prediction..."
    conda activate convnext-mhc
    cd "$CONVNEXT_DIR/ConvNeXt-MHC"
    
    # Modify run_model.py to use example.csv
    echo "Note: Edit run_model.py to point to your test file"
    echo "Example: data_file_name = 'example.csv'"
fi

echo ""
echo "✓ All done!"
