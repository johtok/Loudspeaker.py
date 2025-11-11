"""Tests for Neural CDE implementations."""

import pytest
import jax
import jax.numpy as jnp
import torch

# Import from our package
try:
    import src.loudspeaker_py.models.neural_cdes as neural_cdes
    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="neural_cdes module not available")
class TestNeuralCDE:
    """Test Neural CDE functionality."""
    
    def test_basic_imports(self):
        """Test that all required imports work."""
        import diffrax
        import equinox as eqx
        import optax
        import jax.random as jr
        
        assert diffrax is not None
        assert eqx is not None
        assert optax is not None
        assert jr is not None
    
    def test_jax_device_availability(self):
        """Test JAX device availability."""
        devices = jax.devices()
        assert len(devices) > 0
        print(f"JAX devices: {devices}")
    
    def test_torch_cuda_availability(self):
        """Test PyTorch CUDA availability."""
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        print(f"PyTorch CUDA available: {cuda_available}")
        print(f"PyTorch device count: {device_count}")
    
    def test_array_operations(self):
        """Test basic array operations in both frameworks."""
        # JAX operations
        jax_array = jnp.array([1.0, 2.0, 3.0])
        jax_result = jnp.sin(jax_array)
        assert jax_result.shape == (3,)
        
        # PyTorch operations
        torch_array = torch.tensor([1.0, 2.0, 3.0])
        torch_result = torch.sin(torch_array)
        assert torch_result.shape == (3,)
        
        print("✅ Array operations work in both frameworks")
    
    def test_basic_model_structure(self):
        """Test basic model structure if available."""
        if IMPORT_SUCCESS:
            # Test that we can access the module
            assert hasattr(neural_cdes, '__init__')
            print("✅ Neural CDE module is accessible")
        else:
            pytest.skip("Neural CDE module not available")


class TestEnvironmentSetup:
    """Test environment setup and dependencies."""
    
    def test_required_packages(self):
        """Test that all required packages are available."""
        required_packages = [
            'numpy', 'scipy', 'matplotlib',
            'torch', 'tensorboard',
            'jax', 'diffrax', 'equinox', 'optax',
            'jupyter', 'ipykernel', 'ipywidgets'
        ]
        
        missing_packages = []
        
        for package_name in required_packages:
            try:
                __import__(package_name)
                print(f"✅ {package_name}")
            except ImportError:
                print(f"❌ {package_name}")
                missing_packages.append(package_name)
        
        if missing_packages:
            pytest.fail(f"Missing packages: {missing_packages}")
    
    def test_version_compatibility(self):
        """Test version compatibility of key packages."""
        import sys
        assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version}"
        
        import torch
        assert torch.__version__ >= '2.0.0', f"PyTorch 2.0+ required, got {torch.__version__}"
        
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
        print(f"✅ PyTorch {torch.__version__}")


def test_basic_functionality():
    """Basic functionality test."""
    # Test JAX
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (10,))
    y = jnp.sin(x)
    assert y.shape == (10,)
    
    # Test PyTorch
    a = torch.randn(10)
    b = torch.sin(a)
    assert b.shape == torch.Size([10])
    
    print("✅ Basic functionality tests passed")


if __name__ == "__main__":
    # Run basic tests when script is executed directly
    test_basic_functionality()
    print("All basic tests passed!")