import subprocess
import sys
from pathlib import Path


def run_training_example():
    
    print("=" * 60)
    print("Training Example: Small model on synthetic data")
    print("=" * 60)
    
    cmd = [
        sys.executable, "ixlinx_hack.py", "train",
        "--epochs", "1",
        "--max-meta-tasks", "10",
        "--synthetic",
        "--output-dir", "./outputs",
        "--dim", "128",
        "--layers", "4",
        "--rmc-hidden", "256",
        "--low-rank", "64",
        "--support-size", "2",
        "--query-size", "2",
        "--inner-steps-max", "2",
        "--log-interval", "2",
        "--verbosity", "info",
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def run_evaluation_example():
    
    print("\n" + "=" * 60)
    print("Evaluation Example: Testing the checkpoint")
    print("=" * 60)
    
    checkpoint_path = "./outputs/ixlinx_hack.ckpt"
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Run training first!")
        return False
    
    cmd = [
        sys.executable, "ixlinx_hack.py", "eval",
        "--checkpoint", checkpoint_path,
        "--max-meta-tasks", "5",
        "--synthetic",
        "--verbosity", "info",
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def run_chat_example():
    
    print("\n" + "=" * 60)
    print("Chat Example: Interactive generation")
    print("=" * 60)
    
    checkpoint_path = "./outputs/ixlinx_hack.ckpt"
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Run training first!")
        return False
    
    cmd = [
        sys.executable, "ixlinx_hack.py", "chat",
        "--checkpoint", checkpoint_path,
        "--prompt", "Hello, this is a test of the iXlinx model!",
        "--max-new-tokens", "20",
        "--temperature", "0.9",
        "--verbosity", "info",
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def run_config_export_example():
    
    print("\n" + "=" * 60)
    print("Config Export Example")
    print("=" * 60)
    
    cmd = [
        sys.executable, "ixlinx_hack.py", "export-config",
        "--output", "ixlinx_config.json",
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        config_path = Path("ixlinx_config.json")
        if config_path.exists():
            print(f"\nConfig exported to: {config_path}")
            print("First 10 lines:")
            with open(config_path) as f:
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    print(f"  {line.rstrip()}")
            print("  ...")
    
    return result.returncode == 0


def main():
    
    print("iXlinx-8B Prototype - Example Usage")
    print("=" * 60)
    print("This script demonstrates the key features of the ixlinx_hack.py")
    print("implementation. Each step runs a minimal example.\n")
    
    examples = [
        ("Export Config", run_config_export_example),
        ("Training", run_training_example),
        ("Evaluation", run_evaluation_example),
        ("Chat/Generation", run_chat_example),
    ]
    
    results = {}
    for name, func in examples:
        try:
            success = func()
            results[name] = "✓ PASSED" if success else "✗ FAILED"
        except Exception as e:
            print(f"\n✗ Example '{name}' raised an exception: {e}")
            results[name] = "✗ ERROR"
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, result in results.items():
        print(f"{name:20s}: {result}")
    
    all_passed = all("PASSED" in r for r in results.values())
    print("\n" + ("All examples passed! ✓" if all_passed else "Some examples failed."))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
