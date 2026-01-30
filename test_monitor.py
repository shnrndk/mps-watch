from src.mps_watch import monitor
import torch
import time

print(f"MPS Available: {torch.backends.mps.is_available()}")

@monitor("Test Decorator")
def run_decorator():
    if torch.backends.mps.is_available():
        x = torch.ones(1024*1024*10, device="mps") # ~40MB
        time.sleep(0.5)
        del x

def run_context():
    with monitor("Test Context"):
        if torch.backends.mps.is_available():
            y = torch.ones(1024*1024*20, device="mps") # ~80MB
            time.sleep(0.5)
            del y

if __name__ == "__main__":
    print("--- Running Decorator Test ---")
    run_decorator()
    print("\n--- Running Context Manager Test ---")
    run_context()
    
    print("\n--- Running Direct API Test ---")
    from mps_watch import get_current_memory_usage, get_system_memory
    allocated, reserved = get_current_memory_usage()
    print(f"Current Process MPS Usage -> Allocated: {allocated/1024**2:.2f} MB, Reserved: {reserved/1024**2:.2f} MB")
    
    sys_mem = get_system_memory()
    print(f"System Unified Memory -> Total: {sys_mem['total']/1024**3:.2f} GB, Used: {sys_mem['used']/1024**3:.2f} GB, Pressure: {sys_mem['percent']}%")
