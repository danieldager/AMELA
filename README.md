Linux Requirements
Clang (e.g. 6.0.0-1ubuntu2 verified)
CMake
Terminal
Note that if you did not install libc++1, you have to run the code below to install it:
sudo apt update
sudo apt install libc++1

installing textlesslib

conda create -n textless python=3.9 -y
conda activate textless

git clone git@github.com:facebookresearch/textlesslib.git
pip install -e textlesslib/

pip install git+https://github.com/pytorch/fairseq.git@dd106d9534b22e7db859a6b87ffd7780c38341f8

pip install --no-deps "omegaconf==2.2.3" "hydra-core==1.1.2" "fairseq==0.12.2"
pip install --no-deps "antlr4-python3-runtime==4.9.3"
pip install --no-deps bitarray

pip install git+https://github.com/TEN-framework/ten-vad.git@aa96832d58a295d97b9a6baa4109a9bede4474f8  
