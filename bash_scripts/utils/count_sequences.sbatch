#!/bin/bash
#SBATCH --job-name=count_sequences
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=bash_scripts/logs/count_sequences_%j.log

source bash_scripts/common_setup.sh

bash bash_scripts/count_sequences.sh
