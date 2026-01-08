#!/bin/bash

echo "=================================="
echo "      GPU STRESS & MEMORY TEST    "
echo "=================================="

# 1. Basic Driver Check
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Driver not installed?"
    exit 1
fi

# 2. Run the Python Stress Test in the background
echo "Starting Python Load Generator (30 seconds)..."
# Check if python3 and torch are available
if /work/users/p/r/praful/lib/anaconda/conda/bin/python -c "import torch" &> /dev/null; then
    /work/users/p/r/praful/lib/anaconda/conda/bin/python gpu_stress.py &
    PID=$!
else
    echo "ERROR: PyTorch not installed. Cannot generate load."
    echo "Run: pip install torch"
    exit 1
fi

# 3. Monitor Loop
echo "Monitoring GPU usage (CTRL+C to stop early)..."
echo "------------------------------------------------"

# Run for approx 30 seconds (monitoring every 1s)
for i in {1..30}; do
    # Check if process is still running
    if ! kill -0 $PID 2>/dev/null; then
        break
    fi

    # Print timestamp and specific GPU stats (Util % and Memory)
    echo -ne "$(date +%T) | "
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader
    
    sleep 1
done

wait $PID
echo ""
echo "=================================="
echo "          TEST FINISHED           "
echo "=================================="