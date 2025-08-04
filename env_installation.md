```
# install cuda & cudnn for 4090 workstation
conda create -n verl python==3.10
conda activate verl
git clone https://github.com/volcengine/verl.git
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install -e .
pip install msgspec
cd ..
git clone https://github.com/NVIDIA/apex.git
cd apex
MAX_JOB=64 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..
git clone https://github.com/moojink/openvla-oft.git
git clone https://github.com/PRIME-RL/SimpleVLA-RL.git
cd openvla-oft
vim pyproject.toml # remove installation reqs for torch, torchvision and torchaudio
pip install -e .# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install -r ./experiments/robot/libero/libero_requirements.txt
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
pip install peft==0.11.1 PyOpenGL_accelerate torchdata
conda install -c conda-forge glib libgl glew glfw libglu mesa-libgl-cos6-x86_64 xvfbwrapper -y
conda install conda-forge::libvulkan-loader -y
```