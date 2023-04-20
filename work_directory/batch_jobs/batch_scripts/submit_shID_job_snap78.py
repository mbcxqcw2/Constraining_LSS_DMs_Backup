#!/bin/bash -l

#SBATCH --array=1
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J PYTHON_MP
#SBATCH --mail-type=none
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --partition=fat
#SBATCH --mem=204800 #changed from 512000 to 2048 on 16/07/21

module purge
module load gcc impi/2019.9 #edit: added the /2019.9 after previous errors on 06/01/22
#module load anaconda/3/2019.03
#source /u/cwalker/git_python_downloads/yt-venv/bin/activate
source /u/cwalker/virtual_environments/yt-env/bin/activate
# avoid overbooking of the cores which might occur via NumPy/MKL threading
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export SLURM_HINT=multithread

#include path to python module
#export PYTHONPATH="/ptmp/cwalker/Illustris_FRB_Project/charlie_TNG_lib/":$PYTHONPATH

#echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

srun python ./get_subhaloIDs_1.py TNG300-1 78 70 #$SLURM_ARRAY_TASK_ID #note doubled nodes from 7 -> 14 -> 21 -> 28 -> 42 -> 56 -> 70 on 26/04/22 alongside script change from loading simulation in chunks (doubled from 100 -> 200 -> 300...). If this still works, perhaps double again. If it freezes, change back and investigate further. 