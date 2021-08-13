#!/bin/bash

#SBATCH --job-name=vmc
#SBATCH --account=q-next-systems
#SBATCH --partition=bdwd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --output=vmc.out
#SBATCH --error=vmc.error
#SBATCH --mail-user=dchen@anl.gov
#SBATCH --mail-type=ALL
#SBATCH --time=3:00:00

# Run My Program
conda init bash
conda activate vmc_env
srun python main.py --pb_type=maxindp --fr="NES" --input_size 150 --model_name="rbm" --optimizer="sgd" --batch_size=4000 --learning_rate=0.05 --num_of_iterations=150 --random_seed=666 --penalty=2 --cvar=100 --save_file='cvar'