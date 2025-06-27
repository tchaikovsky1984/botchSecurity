#!/bin/bash

# Define the base directory
BASE_DIR="dataset"

# Check if N (number of classes) is provided as a command-line argument
if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_classes>"
    echo "Example: $0 10"
    exit 1
fi

# Get the number of classes from the first command-line argument
NUM_CLASSES=$1

# Input validation: Ensure NUM_CLASSES is a positive integer
if ! [[ "$NUM_CLASSES" =~ ^[0-9]+$ ]] || [ "$NUM_CLASSES" -le 0 ]; then
    echo "Error: Number of classes must be a positive integer."
    exit 1
fi

# Create the base dataset directory
mkdir -p "$BASE_DIR"

# Loop to create class directories and their subdirectories
for (( i=0; i<NUM_CLASSES; i++ )); do
    CLASS_DIR="$BASE_DIR/class$i"
    mkdir -p "$CLASS_DIR/audio"
    mkdir -p "$CLASS_DIR/images"
done

echo "File structure generated successfully in '$BASE_DIR/'"
echo "Created $NUM_CLASSES class directories (class0 to class$((NUM_CLASSES-1)))."

# To verify the structure, you can run:
tree "$BASE_DIR"
