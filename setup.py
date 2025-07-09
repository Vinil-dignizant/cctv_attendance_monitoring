import os
import platform
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

class PostInstallCommand:
    """Shared logic for install and develop commands"""
    def run(self):
        # Install PyTorch first with appropriate CUDA version
        self.install_torch()
        
        # Install remaining requirements
        self.install_requirements()
        
        # Run original command
        if hasattr(super(), 'run'):
            super().run()

    def install_torch(self):
        """Install PyTorch with appropriate CUDA support"""
        cuda_version = self.detect_cuda()
        
        if cuda_version == "cu121":
            index_url = "https://download.pytorch.org/whl/cu121"
            torch_pkgs = [
                "torch==2.1.0+cu121",
                "torchvision==0.16.0+cu121"
            ]
        elif cuda_version == "cu118":
            index_url = "https://download.pytorch.org/whl/cu118"
            torch_pkgs = [
                "torch==2.1.0+cu118",
                "torchvision==0.16.0+cu118"
            ]
        else:
            index_url = "https://download.pytorch.org/whl/cpu"
            torch_pkgs = [
                "torch==2.1.0+cpu",
                "torchvision==0.16.0+cpu"
            ]
        
        print(f"Installing PyTorch with CUDA: {cuda_version or 'CPU'}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--index-url", index_url] + torch_pkgs
        )

    def detect_cuda(self):
        """Detect CUDA version if available"""
        try:
            # Try nvidia-smi first
            try:
                smi_output = subprocess.check_output(["nvidia-smi"]).decode()
                if "CUDA Version" in smi_output:
                    version_line = [line for line in smi_output.split('\n') if "CUDA Version" in line][0]
                    version = version_line.split("CUDA Version: ")[1].split(" ")[0]
                    return f"cu{version.replace('.', '')[:3]}"
            except:
                pass
            
            # Fallback to nvcc
            output = subprocess.check_output(["nvcc", "--version"]).decode()
            if "release" in output:
                version_line = [line for line in output.split('\n') if "release" in line][0]
                version = version_line.split("release ")[1].split(",")[0]
                return f"cu{version.replace('.', '')[:3]}"
        except:
            pass
        
        # Check environment variables
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home and os.path.exists(cuda_home):
            # Try to detect version from cuda_home
            version_file = os.path.join(cuda_home, "version.txt")
            if os.path.exists(version_file):
                with open(version_file) as f:
                    version = f.read().strip().split()[-1]
                    return f"cu{version.replace('.', '')[:3]}"
            return "cu118"  # Default to CUDA 11.8 if version can't be determined
        return None

    def install_requirements(self):
        """Install all other requirements"""
        is_windows = platform.system() == "Windows"
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # Read requirements.txt
        with open("requirements.txt") as f:
            requirements = [
                line.strip() for line in f 
                if line.strip() and not line.startswith('#')
                and not any(line.startswith(pkg) for pkg in ["torch", "torchvision"])
            ]
        
        # Add platform-specific packages
        if is_windows:
            requirements.extend([
                "pywin32>=300;python_version>='3.6'",
                "pypiwin32;python_version<'3.6'"
            ])
        
        # Handle lap package differently for Windows + Python 3.12+
        if is_windows and float(python_version) >= 3.12:
            requirements.append("lap==0.4.0")
        else:
            requirements.append("lap>=0.4.0")
        
        # Install all requirements
        print("Installing base requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U"] + requirements)

class CustomDevelopCommand(PostInstallCommand, develop):
    """Custom develop command that handles dependencies"""
    pass

class CustomInstallCommand(PostInstallCommand, install):
    """Custom install command that handles dependencies"""
    pass

setup(
    name="face_recognition_system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-camera face recognition system",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    license="MIT",
    cmdclass={
        "develop": CustomDevelopCommand,
        "install": CustomInstallCommand,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        # Core requirements (excluding torch/torchvision)
        "fastapi==0.103.1",
        "uvicorn==0.23.2",
        "python-dotenv==1.0.0",
        "psycopg2-binary==2.9.7",
        "pyyaml==6.0.1",
        "pytz==2023.3",
        "loguru==0.7.0",
        "sqlalchemy==2.0.23",
        "greenlet==3.0.1",
        
        # Computer Vision
        "opencv-python==4.8.1.78",
        "numpy==1.23.5",
        "scipy==1.11.3",
        "onnxruntime==1.16.1",
        
        # Tracking & Detection
        "scikit-learn==1.3.1",
        "filterpy==1.4.5",
        "scikit-image==0.22.0",
        "lap==0.4.0",
        
        # Database
        "asyncpg==0.28.0",
        
        # Development
        "pytest==7.4.2",
        "pytest-asyncio==0.21.1",
        
        # Packaging
        "pyinstaller==6.2.0",
    ],
)