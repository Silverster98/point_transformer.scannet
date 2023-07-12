#!/bin/bash
#SBATCH -J PointTrans
#SBATCH --comment "PointTrans Semantics Segmentation"

#SBATCH --qos level0
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=9:59:59

#SBATCH -o "/scratch/wangzan/Projects/point_transformer.scannet/outputs/slurm/%x.%j.out"
#SBATCH -e "/scratch/wangzan/Projects/point_transformer.scannet/outputs/slurm/%x.%j.err"

EXP_NAME=$1

python scripts/train_partial_scene.py --tag=${EXP_NAME} --num_workers=4 --use_color --npoint 8192