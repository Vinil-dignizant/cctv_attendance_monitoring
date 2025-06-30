import os
import platform
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

# Detect system configuration
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"

def detect_cuda():
    """Detect CUDA version if available"""
    try:
        # Try nvcc first
        output = subprocess.check_output(["nvcc", "--version"]).decode()
        if "release" in output:
            version_line = [line for line in output.split('\n') if "release" in line][0]
            version = version_line.split("release ")[1].split(",")[0]
            return f"cu{version.replace('.', '')[:3]}"  # e.g., 11.8 -> cu118
    except:
        pass
    
    # Fallback to checking environment variables
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.exists(cuda_home):
        return "cu118"  # Default to CUDA 11.8 if CUDA exists but version detection failed
    
    return None

def get_torch_installation():
    """Determine appropriate PyTorch installation string"""
    cuda_version = detect_cuda()
    
    if cuda_version == "cu121":
        return [
            "torch==2.1.0+cu121",
            "torchvision==0.16.0+cu121",
            "--index-url https://download.pytorch.org/whl/cu121"
        ]
    elif cuda_version == "cu118":
        return [
            "torch==2.1.0+cu118",
            "torchvision==0.16.0+cu118",
            "--index-url https://download.pytorch.org/whl/cu118"
        ]
    else:
        return [
            "torch==2.1.0+cpu",
            "torchvision==0.16.0+cpu",
            "--index-url https://download.pytorch.org/whl/cpu"
        ]

class CustomInstall(install):
    """Custom installation that handles platform-specific dependencies"""
    def run(self):
        # Install PyTorch first
        torch_requirements = get_torch_installation()
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + torch_requirements)
        
        # Install platform-specific packages
        extra_packages = []
        
        if IS_WINDOWS:
            extra_packages.extend([
                "pywin32>=300;python_version>='3.6'",
                "pypiwin32;python_version<'3.6'"
            ])
        
        # Install lap package conditionally
        if IS_WINDOWS and float(PYTHON_VERSION) >= 3.12:
            extra_packages.append("lap==0.4.0")
        else:
            extra_packages.append("lap>=0.4.0")
        
        # Install all requirements
        with open("requirements.txt") as f:
            requirements = f.read().splitlines()
        
        # Filter out torch/torchvision if already installed
        requirements = [
            req for req in requirements 
            if not req.startswith(("torch==", "torchvision=="))
        ]
        
        all_requirements = requirements + extra_packages
        
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + all_requirements)
        
        # Proceed with normal installation
        install.run(self)

# Package metadata
setup(
    name="face_recognition_system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-camera face recognition system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    cmdclass={
        "install": CustomInstall,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)