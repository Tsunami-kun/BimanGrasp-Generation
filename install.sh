echo "========================================="
echo "BimanGrasp Installation Script"
echo "========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
conda create -n bimangrasp python=3.8 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bimangrasp

# Install PyTorch and CUDA 11.8 dependencies
echo "Installing PyTorch and CUDA toolkit..."
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch3D dependencies
echo "Installing PyTorch3D and dependencies..."
pip install https://github.com/facebookresearch/pytorch3d/archive/refs/tags/V0.7.8.tar.gz

# Install other conda packages
echo "Installing additional packages..."
conda install -c conda-forge transforms3d trimesh plotly rtree -y

# Install pip packages
echo "Installing pip packages..."
pip install urdf_parser_py scipy networkx tensorboard six

# Install TorchSDF
cd thirdparty/TorchSDF
bash install.sh
cd ../..

# Install modified pytorch_kinematic
cd thirdparty/pytorch_kinematics
pip install -e .
cd ../..

echo "========================================="
echo "Installation complete!"
echo "To activate the environment, run:"
echo "  conda activate bimangrasp"
echo "========================================="
