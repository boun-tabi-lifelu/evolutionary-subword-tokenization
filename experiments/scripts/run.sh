#!/bin/bash
#SBATCH --job-name=run
#SBATCH --mail-type=ALL
#SBATCH --output=%j.log
#SBATCH --mail-user=username@gmail.com
#SBATCH --container-image ghcr.io\#bouncmpe/cuda-python3
#SBATCH --container-mounts /home/user/:/home/user/
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=10G

echo $1

source /opt/python3/venv/base/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu118
cd ~/template_repo
pip install -e . 

python experiments/run.py --config-name $1