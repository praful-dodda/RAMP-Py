import torch
import time
import sys

def run_stress_test(duration=30):
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available in PyTorch.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"Generated load on: {gpu_name}")
    
    # --- 1. MEMORY TEST (VRAM Allocation) ---
    print("Attempting to allocate ~2GB of VRAM...")
    try:
        # Float32 is 4 bytes. 500 million elements * 4 bytes = ~2GB
        buffer = torch.randn(500_000_000, dtype=torch.float32, device='cuda')
        print("SUCCESS: 2GB VRAM allocated.")
    except RuntimeError as e:
        print(f"WARNING: Could not allocate 2GB ({e}). Trying 500MB...")
        buffer = torch.randn(125_000_000, dtype=torch.float32, device='cuda')

    # --- 2. COMPUTE TEST (Keep GPU at 100% util) ---
    print(f"Running matrix multiplications for {duration} seconds...")
    
    # Create two matrices for multiplication
    size = 4096 # Large enough to saturate cores
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')

    start_time = time.time()
    loops = 0
    
    while (time.time() - start_time) < duration:
        # Perform matrix multiplication
        c = torch.matmul(a, b)
        # Synchronize to ensure GPU actually finishes the work before looping
        torch.cuda.synchronize()
        loops += 1

    print(f"Test Complete. Performed {loops} large matrix multiplications.")
    print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    run_stress_test()