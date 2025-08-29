#!/bin/bash

# Directory containing the files to push
BATCH_DIR="small_batch/batch-000"

# Short pause duration in seconds (e.g., 0.5 seconds)
SLEEP_TIME=0.5

# Loop over each file in the directory
for file in "$BATCH_DIR"/*; do
    if [ -f "$file" ]; then
        git add "$file"
        git commit -m "Add file $(basename "$file")"
        git push origin main  # Replace 'main' with your branch if needed
        echo "File $(basename "$file") added, committed, and pushed."
        
        # Short pause
        sleep $SLEEP_TIME
    fi
done

echo "All files in $BATCH_DIR have been pushed."