# Installation
'
conda create -n finetune python=3.10 -y
conda activate finetune
python -m pip install --upgrade pip

# For CUDA 12.4.
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Install tiny-cuda-nn
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install -r requirements.txt

'