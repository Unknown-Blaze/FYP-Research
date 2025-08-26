#!/bin/bash

# Step 1: Source the .env file to load all directory variables into the environment.
if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
else
    echo "Error: .env file not found. Please create it."
    exit 1
fi

# Step 2: Build the string of command-line arguments from parameters.txt
PARAMS=""
while IFS='=' read -r key value || [ -n "$key" ]; do
    # Trim whitespace from key and value
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)
    
    # Skip comments and empty lines
    if [[ ! $key =~ ^# && -n $key ]]; then
        # Special handling for train_all parameter which always needs a value
        if [[ "$key" == "train_all" ]]; then
            PARAMS+="--$key $value "
        # Check for boolean flags for other parameters
        elif [[ "$value" == "yes" ]]; then
            PARAMS+="--$key "
        elif [[ "$value" == "no" ]]; then
            :
        else
            PARAMS+="--$key $value "
        fi
    fi
done < parameters.txt

# Step 3: Execute the python script, passing BOTH the optional parameters
# AND the required positional arguments for the directories.
# The order of positional arguments must match the script's usage string.
python epi_graph.py $PARAMS \
    "params_from_script" \
    "$DIRECTORY_DATA" \
    "$DIRECTORY_PROCESSED_DATA" \
    "$DIRECTORY_PDB_LIST" \
    "$DIRECTORY_UNSEEN_PDB_LIST" \
    "$DIRECTORY_AF_PDB_LIST" \
    "$DIRECTORY_UNSEEN_AF_PDB_LIST" \
    "$DIRECTORY_OUTPUT"
