#! /bin/bash

# Send output to folder, time, memory, nodes
#SBATCH --output /home/rbouma/Thesis/Scripts/output/stdout.txt
#SBATCH --error /home/rbouma/Thesis/Scripts/output/stderr.txt
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH -N 1 -n 6



/home/rbouma/venv/gdsm/bin/python3 /home/rbouma/Thesis/Scripts/Nocleansim.py 
