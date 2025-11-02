#!/bin/bash
#
# ColabFold MHC-Peptide Structure Generation Runner
# Optimized for A100 GPU with batch processing
#

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# ColabFold environment setup
COLABFOLD_DIR="$HOME/FYP-Research/ESM2_Pipeline/structure/localcolabfold"
PYTHON="${COLABFOLD_DIR}/colabfold-conda/bin/python"
COLABFOLD_LIB="${COLABFOLD_DIR}/colabfold-conda/lib"

# Script and data paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="${SCRIPT_DIR}/../iedb_dataset.csv"
OUTPUT_DIR="${SCRIPT_DIR}/data/generated_pdbs"

# Batch processing parameters
BATCH_SIZE=50              # Sequences per batch (increase if more VRAM available)
MAX_SAMPLES=10           # null = all; set to number to limit
START_IDX=0                # Start index for resumable runs

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "$1"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
}

print_info() {
    echo "[INFO] $1"
}

print_error() {
    echo "[ERROR] $1" >&2
}

check_environment() {
    print_header "ENVIRONMENT CHECK"
    
    # Check Python
    if [ ! -f "$PYTHON" ]; then
        print_error "Python not found at: $PYTHON"
        exit 1
    fi
    print_info "✓ Python: $PYTHON"
    
    # Check CSV input
    if [ ! -f "$CSV_FILE" ]; then
        print_error "CSV file not found at: $CSV_FILE"
        exit 1
    fi
    print_info "✓ CSV: $CSV_FILE"
    
    # Check ColabFold library
    if [ ! -d "$COLABFOLD_LIB" ]; then
        print_error "ColabFold library not found at: $COLABFOLD_LIB"
        exit 1
    fi
    print_info "✓ ColabFold library: $COLABFOLD_LIB"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    print_info "✓ Output directory: $OUTPUT_DIR"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    print_header "COLABFOLD MHC-PEPTIDE STRUCTURE GENERATION"
    
    # Check environment
    check_environment
    
    # Set library path for GLIBC compatibility
    export LD_LIBRARY_PATH="${COLABFOLD_LIB}:${LD_LIBRARY_PATH}"
    print_info "✓ LD_LIBRARY_PATH configured"
    
    # Define the input FASTA we just created
    INPUT_FASTA="${SCRIPT_DIR}/colabfold_input.fasta"
    
    if [ ! -f "$INPUT_FASTA" ]; then
        print_error "Input FASTA not found: $INPUT_FASTA"
        print_error "Please run 'python prepare_fasta.py' first."
        exit 1
    fi
    
    print_info "Configuration:"
    echo "  Input: $INPUT_FASTA"
    echo "  Output: $OUTPUT_DIR"
    echo ""
    
    # Run prediction
    print_header "STARTING STRUCTURE GENERATION"
    
    # This is the optimized command.
    # The --templates false line has been REMOVED.
    CUDA_VISIBLE_DEVICES=2 colabfold_batch \
      --num-recycle 3 \
      --num-models 1 \
      --model-type alphafold2_ptm \
      "$INPUT_FASTA" \
      "$OUTPUT_DIR"
    
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        print_header "✓ GENERATION COMPLETE"
    else
        print_error "Structure generation failed with exit code $RESULT"
        exit 1
    fi
}

# ============================================================================
# HELP AND USAGE
# ============================================================================

show_help() {
    cat << EOF
ColabFold MHC-Peptide Structure Generation Runner

USAGE:
    ./run_colabfold.sh [OPTIONS]

OPTIONS:
    --batch-size N      Set batch size (default: 50)
                        Use 100-200 for more VRAM
    
    --max-samples N     Limit number of structures (default: all)
                        Useful for testing
    
    --start-idx N       Start from index N (default: 0)
                        Useful for resuming interrupted runs
    
    --help              Show this message

EXAMPLES:
    # Standard run (all data, batch size 50)
    ./run_colabfold.sh
    
    # Test with 100 samples, larger batches
    ./run_colabfold.sh --batch-size 100 --max-samples 100
    
    # Resume from index 5000 with batch size 200
    ./run_colabfold.sh --batch-size 200 --start-idx 5000

VRAM OPTIMIZATION:
    Batch size 50:   ~4-6 GB VRAM per batch   (default)
    Batch size 100:  ~8-12 GB VRAM per batch  (recommended for A100 80GB)
    Batch size 200:  ~16-24 GB VRAM per batch (for A100 with 30GB limit)
    
    Edit BATCH_SIZE in this script to change default.

ESTIMATED RUNTIME:
    - First run: ~2 minutes (model loading + initialization)
    - Subsequent batches: ~1.5-2 min per 50 sequences on A100
    - Full 85k dataset: ~42-56 hours with batch size 50
    - With batch size 200: ~10-14 hours
EOF
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --start-idx)
            START_IDX="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main