#! /bin/bash

# Send output to folder, time, memory, nodes
#SBATCH --output /home/rbouma/Thesis/Scripts/output/stdout_h.txt
#SBATCH --error /home/rbouma/Thesis/Scripts/output/stderr_h.txt
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
#SBATCH -N 1 -n 4



/home/rbouma/venv/gdsm/bin/python3 /home/rbouma/Thesis/Scripts/Nocleansim.py 3 
