#!/bin/bash
#SBATCH -JHPPC
#SBATCH -A gts-awu36-paid
#SBATCH -N1 --ntasks=1 --cpus-per-task=12
#SBATCH --gres=gpu:A100:1 --mem-per-gpu=96G --exclusive
#SBATCH -t 15:00:00
#SBATCH -qinferno
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=splion360@gmail.com
#SBATCH -o run_n32.out

cd ~/p-awu36-0/HPPC/
module load pytorch
python main.py --num-workers 8 --num-envs 4