#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=Evaluate_BiGRU_model_conj
#SBATCH --mail-type=END
#SBATCH --mail-user=zg475@nyu.edu
#SBATCH --output=slurm_%j.out

NAME="DiscSentEmbed-Evaluation"
WORKDIR="${SCRATCH}/${NAME}"
module purge
module load python/intel/2.7.12 pytorch/0.2.0_1 protobuf/intel/3.1.0
module load scikit-learn/intel/0.18.1

PYTHONPATH=$PYTHONPATH:. python Evaluate.py   -data="./Data/GutenWiki" -model="model_2017-12-17_021305_731265"   -sf='_3' -aeb