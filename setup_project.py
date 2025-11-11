#!/usr/bin/env python3
"""
Loudspeaker Py - Complete Project Setup
=======================================

This script sets up the complete Loudspeaker Py CFES framework environment,
including dependencies, Jupyter kernel, and verification of all components.
"""

import subprocess
import sys
import os
import importlib
from pathlib import Path


def run_command(command, description=""):
    """Run a shell command and return success status."""
    print(f"Running: {command}")
    if description:
        print(f"Description: {description}")
    
    try:
        result = subprocess.run(
            command, shell=True, check=True, 
            capture_output=True, text=True
        )
        print(f"✅ Success: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check Python version compatibility."""
    print("Checking Python version...")
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_pixi():
    """Check if pixi is installed."""
    print("Checking Pixi installation...")
    try:
        result = subprocess.run(["pixi", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Pixi version: {result.stdout.strip()}")
            return True
        else:
            print("❌ Pixi not found")
            return False
    except FileNotFoundError:
        print("❌ Pixi not found")
        return False


def install_dependencies():
    """Install project dependencies using pixi."""
    print("Installing dependencies with Pixi...")
    
    if not run_command("pixi install", "Install all dependencies from pixi.toml"):
        print("❌ Failed to install dependencies")
        return False
    
    print("✅ Dependencies installed successfully")
    return True


def setup_jupyter():
    """Setup Jupyter kernel."""
    print("Setting up Jupyter environment...")
    
    if not run_command("python scripts/setup_jupyter_kernel.py", "Setup Jupyter kernel"):
        print("❌ Failed to setup Jupyter kernel")
        return False
    
    print("✅ Jupyter kernel setup completed")
    return True


def verify_imports():
    """Verify all required packages can be imported."""
    print("Verifying package imports...")
    
    required_packages = [
        # Core scientific packages
        ('numpy', None),
        ('scipy', None),
        ('matplotlib', None),
        
        # Deep learning frameworks
        ('torch', 'torch.__version__'),
        ('torchvision', None),
        
        # JAX ecosystem
        ('jax', 'jax.devices()'),
        ('jaxlib', None),
        
        # Diffrax ecosystem
        ('diffrax', 'diffrax.__version__'),
        ('equinox', 'equinox.__version__'),
        ('optax', 'optax.__version__'),
        
        # Jupyter ecosystem
        ('jupyter', None),
        ('jupyterlab', None),
        ('ipykernel', None),
        ('ipywidgets', None),
        
        # Development tools
        ('pytest', None),
        ('black', None),
        ('flake8', None),
        
        # Utilities
        ('tensorboard', None),
        ('tqdm', None),
        ('pandas', None),
    ]
    
    failed_imports = []
    
    for package_name, version_attr in required_packages:
        try:
            package = importlib.import_module(package_name)
            
            if version_attr:
                try:
                    version = eval(f"package.{version_attr}")
                    print(f"✅ {package_name}: {version}")
                except:
                    print(f"✅ {package_name}: OK")
            else:
                print(f"✅ {package_name}: OK")
                
        except ImportError as e:
            print(f"❌ {package_name}: {e}")
            failed_imports.append(package_name)
    
    if failed_imports:
        print(f"❌ Failed imports: {failed_imports}")
        return False
    
    print("✅ All required packages imported successfully")
    return True


def run_tests():
    """Run the test suite."""
    print("Running test suite...")
    
    if not run_command("pixi run test", "Run basic tests"):
        print("❌ Some tests failed")
        return False
    
    print("✅ All tests passed")
    return True


def create_example_notebook():
    """Create an example Jupyter notebook."""
    print("Creating example notebook...")
    
    notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Loudspeaker Py CFES Framework - Getting Started\\n",
    "\\n",
    "Welcome to the Loudspeaker Py CFES framework! This notebook will help you get started with testing Controlled Functional Expansion Systems.\\n",
    "\\n",
    "## Available Examples\\n",
    "\\n",
    "1. **JAX/Diffrax Neural CDE** - `scripts/neural_cde_example.py`\\n",
    "2. **PyTorch CFES** - `scripts/pytorch_cfe_example.py`\\n",
    "3. **Jupyter Setup** - `scripts/setup_jupyter_kernel.py`\\n",
    "\\n",
    "## Quick Test\\n",
    "\\n",
    "Let's verify everything is working:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Test JAX\\n",
    "import jax\\n",
    "import jax.numpy as jnp\\n",
    "print(f\"JAX devices: {jax.devices()}\")\\n",
    "print(f\"JAX backend: {jax.default_backend()}\")\\n",
    "\\n",
    "# Test PyTorch\\n",
    "import torch\\n",
    "print(f\"PyTorch CUDA: {torch.cuda.is_available()}\")\\n",
    "\\n",
    "# Test Diffrax\\n",
    "import diffrax\\n",
    "print(f\"Diffrax version: {diffrax.__version__}\")\\n",
    "\\n",
    "# Test basic operations\\n",
    "x = jnp.array([1.0, 2.0, 3.0])\\n",
    "y = jnp.sin(x)\\n",
    "print(f\"JAX result: {y}\")\\n",
    "\\n",
    "a = torch.tensor([1.0, 2.0, 3.0])\\n",
    "b = torch.sin(a)\\n",
    "print(f\"PyTorch result: {b}\")\\n",
    "\\n",
    "print(\"\\n✅ Environment is ready!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Next Steps\\n",
    "\\n",
    "1. Run the Neural CDE example:\\n",
    "   ```python\\n",
    "   exec(open('scripts/neural_cde_example.py').read())\\n",
    "   ```\\n",
    "   \\n",
    "2. Or run the PyTorch example:\\n",
    "   ```python\\n",
    "   exec(open('scripts/pytorch_cfe_example.py').read())\\n",
    "   ```\\n",
    "\\n",
    "3. Use the CLI interface:\\n",
    "   ```bash\\n",
    "   loudspeaker-cli --help\\n",
    "   ```"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Loudspeaker Python (CFES)",
   "language": "python",
   "name": "loudspeaker-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
    
    try:
        with open("getting_started.ipynb", "w") as f:
            f.write(notebook_content)
        print("✅ Created getting_started.ipynb")
        return True
    except Exception as e:
        print(f"❌ Failed to create notebook: {e}")
        return False


def print_completion_summary():
    """Print completion summary and next steps."""
    print("\n" + "="*60)
    print("🎉 LOUDSPEAKER PY CFES FRAMEWORK SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 What was set up:")
    print("✅ Pixi environment with all dependencies")
    print("✅ Jupyter kernel: 'Loudspeaker Python (CFES)'")
    print("✅ Example scripts with modular sections (#%%)")
    print("✅ Test suite and development tools")
    print("✅ Command-line interface (CLI)")
    print("✅ Documentation and README")
    
    print("\n🚀 Quick start commands:")
    print("1. Start Jupyter Lab:")
    print("   pixi run jupyter-lab")
    print("2. Run Neural CDE example:")
    print("   python scripts/neural_cde_example.py")
    print("3. Run PyTorch CFES example:")
    print("   python scripts/pytorch_cfe_example.py")
    print("4. Use CLI interface:")
    print("   loudspeaker-cli --help")
    print("5. Run tests:")
    print("   pixi run test")
    
    print("\n📁 Project structure:")
    print("├── src/loudspeaker_py/     # Main package")
    print("├── scripts/               # Example scripts")
    print("├── tests/                 # Test suite")
    print("├── examples/              # Example notebooks")
    print("└── docs/                  # Documentation")
    
    print("\n🔗 Key files:")
    print("- README.md: Complete documentation")
    print("- scripts/neural_cde_example.py: JAX/diffrax example")
    print("- scripts/pytorch_cfe_example.py: PyTorch example")
    print("- scripts/setup_jupyter_kernel.py: Jupyter setup")
    print("- getting_started.ipynb: Jupyter notebook")
    
    print("\n🎯 Framework comparison:")
    print("JAX/Diffrax: High performance, excellent autodiff")
    print("PyTorch: Easy to use, mature ecosystem")
    print("Both frameworks support modular execution with #%% sections")
    
    print("\n" + "="*60)


def main():
    """Main setup function."""
    print("Loudspeaker Py CFES Framework - Complete Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_pixi():
        print("❌ Please install Pixi from https://pixi.rs/")
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Setting up Jupyter", setup_jupyter),
        ("Verifying imports", verify_imports),
        ("Running tests", run_tests),
        ("Creating example notebook", create_example_notebook),
    ]
    
    # Execute setup steps
    for step_name, step_function in steps:
        print(f"\n🔧 {step_name}...")
        if not step_function():
            print(f"❌ Setup failed at: {step_name}")
            sys.exit(1)
    
    # Print completion summary
    print_completion_summary()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)