#!/bin/bash

#SBATCH --job-name=vmc
#SBATCH --account=q-next-systems
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --output=vmc.out
#SBATCH --error=vmc.error
#SBATCH --mail-user=dchen@anl.gov
#SBATCH --mail-type=ALL
#SBATCH --time=4:00:00

# Run My Program
mpirun -n 32 python main.py --pb_type=maxindp --fr="NES" --input_size 100  --model_name="rbm" --optimizer="sgd" --batch_size=16000 --learning_rate=0.007 --num_of_iterations=500 --random_seed=666 --penalty=2 --connect_prob=0.2 --cvar=25 --save_file='alpha_1_size_50_p_02' --nchain=32