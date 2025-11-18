#!/usr/bin/env python3
"""
Setup Jupyter Kernel for Loudspeaker Py Environment
===================================================

This script creates a Jupyter kernel for the current environment
and installs it for use in Jupyter notebooks and JupyterLab.
"""

import subprocess
import sys


def setup_jupyter_kernel():
    """Setup Jupyter kernel for the current environment."""
    print("Setting up Jupyter kernel for Loudspeaker Py...")

    # Get the current Python executable
    python_executable = sys.executable
    print(f"Using Python executable: {python_executable}")

    # Create kernel name
    kernel_name = "loudspeaker-py"
    display_name = "Loudspeaker Python (CFES)"

    try:
        # Install the kernel
        print(f"Installing kernel: {kernel_name}")
        subprocess.run(
            [
                python_executable,
                "-m",
                "ipykernel",
                "install",
                "--user",
                "--name",
                kernel_name,
                "--display-name",
                display_name,
            ],
            check=True,
        )

        print(f"✅ Successfully installed Jupyter kernel: {kernel_name}")
        print(f"   Display name: {display_name}")
        print(f"   Available in Jupyter as: {display_name}")

        # List installed kernels to verify
        print("\nInstalled Jupyter kernels:")
        result = subprocess.run(
            [python_executable, "-m", "jupyter", "kernelspec", "list"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Warning: Could not list kernels")
            print(f"Error: {result.stderr}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Jupyter kernel: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def verify_environment():
    """Verify that required packages are available."""
    print("Verifying environment setup...")

    required_packages = [
        "numpy",
        "scipy",
        "matplotlib",
        "torch",
        "tensorboard",
        "jax",
        "diffrax",
        "equinox",
        "optax",
        "jupyter",
        "ipykernel",
        "ipywidgets",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install them using: pixi install")
        return False

    print("\n✅ All required packages are available!")
    return True


def create_sample_notebook():
    """Create a sample Jupyter notebook to test the setup."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Loudspeaker Py CFES Framework\n",
                    "\n",
                    "Welcome to the Loudspeaker Py CFES framework! This notebook demonstrates the basic usage.\n",
                    "\n",
                    "## Quick Start\n",
                    "\n",
                    "```python\n",
                    "import jax\n",
                    "import jax.numpy as jnp\n",
                    "import diffrax\n",
                    "import torch\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "# Check devices\n",
                    'print(f"JAX devices: {jax.devices()}")\n',
                    'print(f"PyTorch CUDA: {torch.cuda.is_available()}")\n',
                    "```\n",
                    "\n",
                    "## Available Examples\n",
                    "\n",
                    "- `scripts/neural_cde_example.py` - Neural CDE with JAX/diffrax\n",
                    "- `scripts/pytorch_cfe_example.py` - PyTorch CFES implementation\n",
                    "- `src/loudspeaker/` - Main package modules\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Test imports\n",
                    "import jax\n",
                    "import jax.numpy as jnp\n",
                    "import diffrax\n",
                    "import torch\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "# Check devices\n",
                    'print(f"JAX devices: {jax.devices()}")\n',
                    'print(f"JAX backend: {jax.default_backend()}")\n',
                    'print(f"PyTorch CUDA available: {torch.cuda.is_available()}")\n',
                    'print(f"PyTorch device count: {torch.cuda.device_count()}")\n',
                    "\n",
                    "# Test basic operations\n",
                    "x = jnp.array([1.0, 2.0, 3.0])\n",
                    "y = jnp.sin(x)\n",
                    'print(f"JAX sin operation: {y}")\n',
                    "\n",
                    "a = torch.tensor([1.0, 2.0, 3.0])\n",
                    "b = torch.sin(a)\n",
                    'print(f"PyTorch sin operation: {b}")\n',
                    "\n",
                    "# Test diffrax\n",
                    'print(f"Diffrax available: {diffrax}")\n',
                    "\n",
                    'print("\\n✅ Environment is ready for CFES development!")',
                ],
                "outputs": [],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Loudspeaker Python (CFES)",
                "language": "python",
                "name": "loudspeaker-py",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    # Write sample notebook
    sample_notebook_path = "example_usage.ipynb"

    try:
        import json

        with open(sample_notebook_path, "w") as f:
            json.dump(notebook_content, f, indent=2)

        print(f"✅ Created sample notebook: {sample_notebook_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to create sample notebook: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("Loudspeaker Py - Jupyter Environment Setup")
    print("=" * 60)

    # Verify environment
    if not verify_environment():
        print("\n❌ Environment verification failed. Please install missing packages.")
        return False

    # Setup Jupyter kernel
    if not setup_jupyter_kernel():
        print("\n❌ Jupyter kernel setup failed.")
        return False

    # Create sample notebook
    create_sample_notebook()

    print("\n" + "=" * 60)
    print("🎉 Jupyter setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start Jupyter Lab: jupyter lab")
    print("2. Open example_usage.ipynb")
    print("3. Select 'Loudspeaker Python (CFES)' kernel")
    print("4. Run the cells to verify everything works")
    print("\nAvailable example scripts:")
    print("- scripts/neural_cde_example.py (JAX/diffrax)")
    print("- scripts/pytorch_cfe_example.py (PyTorch)")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
