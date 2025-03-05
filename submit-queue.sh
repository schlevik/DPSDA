#!/bin/bash

# Configuration
QUEUE_FILE="queue.txt"
PROCESSED_FILE="processed.txt"
LOCK_FILE="/tmp/gpu_queue.lock"
LOG_FILE="logs/gpu_queue.log"
MAX_RETRIES=3
SLEEP_INTERVAL=60  # seconds between checks
export WANDB_PROJECT='synth-data'
mkdir -p logs/
# Function to check for available GPU
find_available_gpu() {
    # Get GPU memory usage using nvidia-smi
    local gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    local gpu_num=0
    
    # Check each GPU's memory usage
    while read -r memory_used; do
        # Consider a GPU available if it's using less than 100MB
        if [ "$memory_used" -lt 100 ]; then
            echo "$gpu_num"
            return 0
        fi
        ((gpu_num++))
    done <<< "$gpu_memory"
    
    return 1  # No available GPU found
}

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Function to get and remove the next command from queue
get_next_command() {
    if [ ! -s "$QUEUE_FILE" ]; then
        return 1
    fi
    head -n 1 "$QUEUE_FILE"
    sed -i '1d' "$QUEUE_FILE"
}

# Function to acquire lock
acquire_lock() {
    if mkdir "$LOCK_FILE" 2>/dev/null; then
        trap 'rm -rf "$LOCK_FILE"' EXIT
        return 0
    fi
    return 1
}

# Main loop
main() {
    local retry_count=0
    
    while true; do
        # Try to acquire lock
        if ! acquire_lock; then
            log_message "Another instance is running. Exiting."
            exit 1
        fi
        
        # Check if queue is empty
        if [ ! -s "$QUEUE_FILE" ]; then
            log_message "Queue is empty. Exiting."
            exit 0
        fi
        
        # Find available GPU
        gpu_id=$(find_available_gpu)
        if [ $? -eq 0 ]; then
            # Get next command from queue
            command=$(get_next_command)
            log_file=logs/`echo $command | cut -d ' ' -f 3,4 | sed s:/:_:g | sed 's: :_:g'`.log
            echo $log_file
            if [ $? -eq 0 ]; then
                # Execute command with CUDA_VISIBLE_DEVICES set
                log_message "Running command on GPU $gpu_id: $command"
                CUDA_VISIBLE_DEVICES=$gpu_id eval "$command" > $log_file 2>&1 &
                
                if [ $? -eq 0 ]; then
                    log_message "Command completed successfully"
                    retry_count=0
                    echo "$command" >> "$PROCESSED_FILE"
                else
                    log_message "Command failed"
                    # Add failed command back to queue
                    echo "$command" >> "$QUEUE_FILE"
                    ((retry_count++))
                fi
            fi
        else
            log_message "No GPU available. Waiting..."
            # ((retry_count++))
        fi
        
        # Release lock
        rm -rf "$LOCK_FILE"
        
        # Check retry count
        if [ $retry_count -ge $MAX_RETRIES ]; then
            log_message "Max retries reached. Exiting."
            exit 1
        fi
        
        # Wait before next check
        sleep "$SLEEP_INTERVAL"
    done
}

# Create log file if it doesn't exist
touch "$LOG_FILE"

# Run main loop
main
