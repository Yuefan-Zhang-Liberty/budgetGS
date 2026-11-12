#!/bin/bash
#SBATCH --nodes=1
#SBATCH --nodelist=node8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --output=/storage/slurm/zyuef/logs/modcheck-%j.out
#SBATCH --error=/storage/slurm/zyuef/logs/modcheck-%j.err
bash /home/stud/zyuef/budgetgs-m0b/modcheck.sh
