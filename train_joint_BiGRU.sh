#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=train_joint_BiGRU
#SBATCH --mail-type=END
#SBATCH --mail-user=zg475@nyu.edu
#SBATCH --output=slurm_%j.out

NAME="DiscSentEmbed"
WORKDIR="${SCRATCH}/${NAME}"
module load python/intel/2.7.12 pytorch/0.2.0_1 protobuf/intel/3.1.0

PYTHONPATH=$PYTHONPATH:. python -u Train.py -order -next -conj -encode='GRU' -bid -in_dim=300  -pre_m="hw" -h_dim=512 -evoc=200000 -data="./Data/GutenWiki"