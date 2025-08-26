#!/usr/bin/env python3

# Ultimate warning suppression script
import warnings
import os
import sys
import logging

# Suppress ALL warnings at the highest level
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Set environment variables to suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# Configure all loggers to ERROR level
logging.basicConfig(level=logging.ERROR)
for name in logging.root.manager.loggerDict:
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    logger.disabled = True

# Suppress specific loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").disabled = True
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch").disabled = True

# Redirect stderr temporarily for CUDA warnings
class NullWriter:
    def write(self, txt): pass
    def flush(self): pass

original_stderr = sys.stderr
sys.stderr = NullWriter()

try:
    # Import torch to get CUDA warnings out of the way
    import torch
    torch.cuda.is_available()
finally:
    # Restore stderr
    sys.stderr = original_stderr

# Set up the command line arguments like the bash script does
import subprocess

# Read the .env file
env_vars = {}
try:
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env_vars[key] = value.strip('"')
except FileNotFoundError:
    print("Warning: .env file not found, using default directory paths")
    # Use default directory paths
    env_vars = {
        'DIRECTORY_DATA': 'data/',
        'DIRECTORY_PROCESSED_DATA': 'data/',
        'DIRECTORY_PDB_LIST': 'lists/',
        'DIRECTORY_UNSEEN_PDB_LIST': 'lists/',
        'DIRECTORY_AF_PDB_LIST': 'lists/',
        'DIRECTORY_UNSEEN_AF_PDB_LIST': 'lists/',
        'DIRECTORY_OUTPUT': 'output/'
    }

# Read parameters.txt and build arguments
params = []
try:
    with open('parameters.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if value == "yes":
                    params.append(f"--{key}")
                elif value != "no":
                    params.extend([f"--{key}", value])
except FileNotFoundError:
    print("Error: parameters.txt file not found")
    sys.exit(1)

# Set up sys.argv as if called from command line
sys.argv = ['epi_graph.py'] + params + [
    "params_from_script",
    env_vars.get('DIRECTORY_DATA', ''),
    env_vars.get('DIRECTORY_PROCESSED_DATA', ''),
    env_vars.get('DIRECTORY_PDB_LIST', ''),
    env_vars.get('DIRECTORY_UNSEEN_PDB_LIST', ''),
    env_vars.get('DIRECTORY_AF_PDB_LIST', ''),
    env_vars.get('DIRECTORY_UNSEEN_AF_PDB_LIST', ''),
    env_vars.get('DIRECTORY_OUTPUT', '')
]

# Now execute the main script
if __name__ == "__main__":
    exec(open('epi_graph.py').read())
