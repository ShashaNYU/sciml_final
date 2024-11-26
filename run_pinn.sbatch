#!/bin/bash
#SBATCH --array=999,536,186,197,987,406,967,204,221,73
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G
#SBATCH --time=3000
#SBATCH -o log/pinn_%A_%a.output
#SBATCH -e log/pinn_%A_%a.error


layers="2 100 100 100 100 2"
N0=50
N_b=50
N_f=20000
num_epoch=50000
seed=$SLURM_ARRAY_TASK_ID
device=0
ft_max_iter=2000
ft_tolerance_grad=1e-6
ft_tolerance_change=1e-7
ft_chunks=5

python pinn.py --layers $layers --N0 $N0 --N_b $N_b --N_f $N_f --num_epoch $num_epoch\
               --seed $seed --device $device --ft_max_iter $ft_max_iter --ft_chunks $ft_chunks\
               --ft_tolerance_grad $ft_tolerance_grad --ft_tolerance_change $ft_tolerance_change